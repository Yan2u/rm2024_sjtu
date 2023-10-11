#ifndef RM_TASK_HPP
#define RM_TASK_HPP

#include <future>
#include <type_traits>
#include <vector>
namespace rm {
class Task {
private:
    // 指针别名
    template<typename T>
    using Ptr = T*;

    // 验证线程函数是否具有指定的参数列表 - 失败
    template<typename Func, typename Arg, typename Return>
    static std::false_type has_valid_args_helper(double) {
        return std::false_type();
    }

    // 验证线程函数是否具有指定的参数列表 - 成功
    template<typename Func, typename Arg, typename Return>
    static auto has_valid_args_helper(int
    ) -> decltype(std::declval<Ptr<Func>>()(std::declval<Ptr<const Arg>>(), std::declval<Ptr<const Arg>>(), std::declval<Ptr<Return>>(), std::declval<Ptr<Return>>()), std::true_type()) {
        return std::true_type();
    }

    // 验证线程函数是否具有指定的参数列表 - 使用 SFINAE 机制
    template<typename Func, typename Arg, typename Return>
    static constexpr bool has_valid_args =
        decltype(Task::has_valid_args_helper<Func, Arg, Return>(0))::value;

    // 验证线程函数是否具有指定的参数列表 - 失败
    template<typename Callable, typename Arg, typename Return>
    static std::false_type has_valid_args_callable_helper(double) {
        return std::false_type();
    }

    // 验证线程函数是否具有指定的参数列表 - 成功
    template<typename Callable, typename Arg, typename Return>
    static auto has_valid_args_callable_helper(int
    ) -> decltype(std::declval<Callable>()(std::declval<Ptr<const Arg>>(), std::declval<Ptr<const Arg>>(), std::declval<Ptr<Return>>(), std::declval<Ptr<Return>>()), std::true_type()) {
        return std::true_type();
    }

    // 验证线程函数是否具有指定的参数列表 - 使用 SFINAE 机制
    template<typename Func, typename Arg, typename Return>
    static constexpr bool has_valid_args_callable =
        decltype(Task::has_valid_args_callable_helper<Func, Arg, Return>(0))::value;

    // 验证传入的 `Func` 是否是返回类型 void 且具有指定参数列表的函数
    template<typename Func, typename Arg, typename Return>
    static constexpr bool is_valid_task_func =
        Task::has_valid_args<Func, Arg, Return>
        && std::is_same_v<
            void,
            std::result_of_t<Ptr<Func>(Ptr<const Arg>, Ptr<const Arg>, Ptr<Return>, Ptr<Return>)>>;

    // 验证传入的 `Func` 是否是返回类型 void 且具有指定参数列表的函数
    template<typename Callable, typename Arg, typename Return>
    static constexpr bool
        is_valid_callable_obj = Task::has_valid_args_callable<Callable, Arg, Return>
        && std::is_same_v<void,
                          decltype(std::declval<Callable>()(
                              std::declval<Ptr<const Arg>>(),
                              std::declval<Ptr<const Arg>>(),
                              std::declval<Ptr<Return>>(),
                              std::declval<Ptr<Return>>()
                          ))>;

public:
    // std::vector<T> 别名
    template<typename T>
    using Array = std::vector<T>;

    /**
     * @brief 开 `task_count` 个线程 (`std::async`) 循环执行 `function`, 阻塞当前线程
     * @note `function` 需要具有签名如: 
     * `void function(const Arg* arg_start, const Arg* arg_end, Return* ret_start, Return* ret_end)`, 
     * 表示对 `[arg_start, arg_end)` 范围内的每个 Arg 元素计算，将结果存在对应位置的 `Return` 指针中.
     * @tparam Func 线程函数类型
     * @tparam Arg 线程函数参数类型
     * @tparam Return 线程函数返回值 (以引用形式入参) 类型
     * @param function 线程函数
     * @param data 参数数组: `std::vector<Arg>`
     * @param results 用来接收结果的数组: `std::vector<Return>`
     * @param task_count 开的线程数
    */
    template<typename Func, typename Arg, typename Return>
    static void run_task_ref_await(
        Func* function,
        const Array<Arg>& data,
        Array<Return>& results,
        int task_count
    ) {
        static_assert(std::is_function_v<Func>, "argument \"function\" must be a function");
        static_assert(
            Task::is_valid_task_func<Func, Arg, Return>,
            "argument \"function\" is not a valid function"
        );

        int n = data.size();
        int each = n / task_count;
        results.resize(n);

        const Arg* data_ptr = data.data();
        Return* results_ptr = results.data();

        if (each < 1) {
            function(data_ptr, data_ptr + n, results_ptr, results_ptr + n);
            return;
        }

        int remain = n - each * task_count;
        int pos = 0;
        std::future<void>* tasks = new std::future<void>[task_count]();
        for (int i = 0; i < task_count; ++i) {
            int each2 = each;
            if (remain > 0) {
                ++each2;
                --remain;
            }

            tasks[i] = std::async(
                std::launch::async,
                function,
                data_ptr + pos,
                data_ptr + pos + each2,
                results_ptr + pos,
                results_ptr + pos + each2
            );
            pos += each2;
        }

        for (int i = 0; i < task_count; ++i) {
            tasks[i].wait();
        }

        delete[] tasks;
    }

    /**
     * @brief 开 `task_count` 个线程 (`std::async`) 循环执行可调用对象 `callable_object`, 阻塞当前线程
     * @note `callable_object` 需要能这样使用: 
     * `callable_object(const Arg* arg_start, const Arg* arg_end, Return* ret_start, Return* ret_end)` 返回值为 void, 
     * 表示对 `[arg_start, arg_end)` 范围内的每个 Arg 元素计算，将结果存在对应位置的 `Return` 指针中.
     * @tparam Callable 可调用对象类型
     * @tparam Arg 线程函数参数类型
     * @tparam Return 线程函数返回值 (以引用形式入参) 类型
     * @param callable_object 可调用对象
     * @param data 参数数组: `std::vector<Arg>`
     * @param results 用来接收结果的数组: `std::vector<Return>`
     * @param task_count 开的线程数
    */
    template<typename Callable, typename Arg, typename Return>
    static void run_task_ref_await(
        Callable& callable_object,
        const Array<Arg>& data,
        Array<Return>& results,
        int task_count
    ) {
        static_assert(
            Task::is_valid_callable_obj<Callable, Arg, Return>,
            "argument \"callable_object\" is not a valid callable object"
        );

        int n = data.size();
        int each = n / task_count;
        results.resize(n);

        const Arg* data_ptr = data.data();
        Return* results_ptr = results.data();

        if (each < 1) {
            callable_object(data_ptr, data_ptr + n, results_ptr, results_ptr + n);
            return;
        }

        int remain = n - each * task_count;
        int pos = 0;
        std::future<void>* tasks = new std::future<void>[task_count]();
        for (int i = 0; i < task_count; ++i) {
            int each2 = each;
            if (remain > 0) {
                ++each2;
                --remain;
            }

            tasks[i] = std::async(
                std::launch::async,
                callable_object,
                data_ptr + pos,
                data_ptr + pos + each2,
                results_ptr + pos,
                results_ptr + pos + each2
            );
            pos += each2;
        }

        for (int i = 0; i < task_count; ++i) {
            tasks[i].wait();
        }

        delete[] tasks;
    }
};
} // namespace rm
#endif