#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel invocation of kwarg functions with arbitrary return types using multiprocessing
"""
__author__ = "Miguel Hernández Cabronero <miguel.hernandez@uab.cat>"
__date__ = "30/07/2019"

import os
import sys
import multiprocessing
import subprocess
import datetime
import traceback
import queue
import random
from pprint import pprint


class ParallelInvoker:
    """Run shell commands or function invocations in parallel.
    """

    def __init__(self, invocation_list, process_count=None):
        """Setup a class tu run list of shell commands (one per line) in parallel.
        Call run() to start the execution.

        :param invocation_list: list of strings (shell commands), or list of (function, kwargs) pairs
        :param process_count: maximum number of processs in parallel (if None, all cores are used)
        """
        self.invocation_list = invocation_list
        self.process_count = process_count \
            if process_count is not None else multiprocessing.cpu_count()

    def run(self, on_result_func=None, exit_on_error=True, verbose=False):
        if verbose:
            print(f"[ParallelInvoker] Running with {self.process_count} "
                  f"process{'es' if self.process_count > 1 else ''}")
        if self.process_count > 1:
            return self.run_parallel(on_result_func=on_result_func,
                                     exit_on_error=exit_on_error,
                                     verbose=verbose)
        else:
            return self.run_sequential(on_result_func=on_result_func,
                                       exit_on_error=exit_on_error,
                                       verbose=verbose)

    last_chars = ["\\", "-", "/", "|"]
    def get_progress_bar(self, index, length, bar_length):

        current_bar_index = int(index*bar_length/length)
        s = "#"*max(0, current_bar_index-1)
        current_char = self.last_chars[index % len(self.last_chars)]
        s += current_char
        s += "-"*(bar_length-current_bar_index if current_bar_index > 0 else bar_length - 1)
        return s

    def run_sequential(self, on_result_func=None, exit_on_error=True, verbose=False):
        """Run all tasks sequentially, and return a list of results.
        :param on_result_func: callable to be invoked for each obtained result (all invoked in a single process).
          It is invoked with (function, kwargs, function(**kwargs)
        :return:
          If invocation_list contains (function, kwargs) tuples, a list of (function, kwargs, result)
          tuples is returned.
          If invocation_list contains a list of shell commands, a list of (shell_command, status, output)
          tuples is returned
        """
        results = []
        failed_invocation_exception = []

        for invocation_index, invocation in enumerate(self.invocation_list):
            percentage = 100*invocation_index / len(self.invocation_list)

            if verbose:
                bar_length = 25
                # os.system('cls' if os.name == 'nt' else 'clear')
                print(f"[ParallelInvoker] Running invocation {invocation_index+1}/{len(self.invocation_list)} "
                      f"({percentage:.1f}%)")
                print("[" + self.get_progress_bar(
                    index=invocation_index, length=len(self.invocation_list), bar_length=bar_length) + "]")

            if type(invocation) == tuple:
                assert len(invocation) == 2
                function, kwargs = invocation
                try:
                    r = function(**kwargs)
                    results.append((function, kwargs, r))
                    if on_result_func is not None:
                        on_result_func(function=function, kwargs=kwargs, result=r)
                except Exception as ex:
                    invocation_msg = "{function_name}(**{args})".format(
                        function_name=function.__name__,
                        args=str({k: v for k, v in kwargs.items() if k != 'input'}))

                    msg = f"Error in invocation {invocation_msg}:\n{type(ex).__name__}: {ex}"
                    if exit_on_error:
                        raise Exception(msg) from ex
                    else:
                        if verbose:
                            print(msg)
                        failed_invocation_exception.append((invocation, ex))

            elif type(invocation) == str:
                invocation = invocation.strip()
                status, output = subprocess.getstatusoutput(invocation)
                results.append((invocation, status, output))
                if on_result_func is not None:
                    on_result_func(invocation=invocation, status=status, output=output)
            else:
                raise TypeError(type(invocation))

        print("[watch] len(failed_invocation_exception) = {}".format(len(failed_invocation_exception)))

        if verbose:
            self.print_failed_invocation_exception_list(failed_invocation_exception)

        return results

    def run_parallel(self, on_result_func=None, exit_on_error=True, verbose=False):
        """Run all tasks in parallel, and return a list of results.
        :param on_result_func: callable to be invoked for each obtained result (all invoked in a single process)
        :return:
          If invocation_list contains (function, kwargs) tuples, a list of (function, kwargs, result)
          tuples is returned.
          If invocation_list contains a list of shell commands, a list of (shell_command, status, output)
          tuples is returned
        """
        # Queue of results (invocation, status, output)
        self.results_queue = multiprocessing.Queue()
        total_jobs = len(self.invocation_list)

        # Queue of pending jobs
        self.pending_jobs = multiprocessing.Queue()
        for invocation in self.invocation_list:
            self.pending_jobs.put(invocation)

        # Pool of Processes
        if verbose:
            print("Starting pool of {} processes...".format(self.process_count))
        process_pool = [ParallelInvoker.Worker(id=i,
                                               pending_jobs_queue=self.pending_jobs,
                                               results_queue=self.results_queue,
                                               verbose=verbose)
                        for i in range(self.process_count)]
        for t in process_pool:
            t.start()

        results = []
        failed_invocation_exception = []

        while len(results) < len(self.invocation_list):
            try:
                r = self.results_queue.get(timeout=2)

                if type(r) == tuple and isinstance(r[0], Exception):
                    if exit_on_error:
                        del self.results_queue
                        raise Exception("Error executing invocation") from r[0]
                    else:
                        failed_invocation_exception.append((r[1], r[0]))
                else:
                    results.append(r)
                    if verbose:
                        # os.system('cls' if os.name == 'nt' else 'clear')
                        print(f"Obtained result #{len(results)}/{len(self.invocation_list)} "
                              f"({100 * len(results) / len(self.invocation_list):.0f}%)"
                              + f"[{self.get_progress_bar(index=len(results),length=len(self.invocation_list), bar_length=25)}]")

                    if on_result_func is not None:
                        if type(r[0]) == str:
                            invocation, status, output = r
                            on_result_func(invocation=invocation, status=status, output=output)
                        else:
                            function, kwargs, result = r
                            on_result_func(function=function, kwargs=kwargs, result=result)
            except queue.Empty:
                if all(not p.is_alive() for p in process_pool):
                    if len(results) + len(failed_invocation_exception) \
                            < len(self.invocation_list):
                        failed_invocation_exception.append(
                            ("<unknown>",
                             Exception("All workers exited without producing the needed output.")))
                        break

        if verbose:
            if len(failed_invocation_exception) > 0:
                print(f"{len(failed_invocation_exception)} failed invocations:")
                pprint(failed_invocation_exception)
            else:
                print()
                print("No invocation failed (-:")
                print()

        return results

    def print_failed_invocation_exception_list(self, invocation_exception_list):
        if len(invocation_exception_list) > 0:
            msg = f"{len(invocation_exception_list)} failed invocations:"
            print()
            print("=" * len(msg))
            print(msg)
            print("=" * len(msg))
            for i, (invocation, exception) in enumerate(invocation_exception_list):
                if type(invocation) == str:
                    invocation_msg = invocation
                elif type(invocation) == tuple:
                    function, kwargs = invocation
                    invocation_msg = "{function_name}(**{args})".format(
                        function_name=function.__name__,
                        args={k: v for k, v in kwargs.items() if k != "input"}
                    )
                print(f"{i:6d}) Invocation: {invocation_msg}:")
                print(f"        {type(exception).__name__}: {exception}")
            print("=" * len(msg))
            print()
        else:
            print()
            print("No invocation failed (-:")
            print()

    class Worker(multiprocessing.Process):
        """ Worker process that draws jobs one at a time, executes it, saves results in the results_queue and
        exists only when pending_jobs_queue is empty.
        """

        def __init__(self, pending_jobs_queue, results_queue, verbose, id=None):
            super().__init__()
            self.results_queue = results_queue
            self.pending_jobs_queue = pending_jobs_queue
            self.verbose = verbose
            self.daemon = True
            self.id = id
            self.current_invocation = None

        def run(self):
            pid_files_dir = "pid_files"
            os.makedirs(pid_files_dir, exist_ok=True)
            pid_file_path = os.path.join("pid_files", f"pid{os.getpid()}")
            try:
                while self.pending_jobs_queue.qsize() > 0:
                    try:
                        open(pid_file_path, "w").write(f"Waiting for next job...\n")

                        self.current_invocation = self.pending_jobs_queue.get(
                            block=True, timeout=1 + random.random() * 0.5)

                        open(pid_file_path, "w").write(f"Executing invocation: {self.current_invocation}...\n")

                        if type(self.current_invocation) == tuple:
                            if self.verbose:
                                pending_count = self.pending_jobs_queue.qsize()
                                print("[ID{} -- ~{} left] {} @ Executing {}({} args)\n".format(
                                    self.ident, pending_count, datetime.datetime.now(),
                                    self.current_invocation[0],
                                    len(self.current_invocation[1])), )

                            assert len(self.current_invocation) == 2
                            function = self.current_invocation[0]
                            kwargs = self.current_invocation[1]
                            r = function(**kwargs)
                            self.results_queue.put((function, kwargs, r))
                        elif type(self.current_invocation) == str:
                            self.current_invocation = self.current_invocation.strip()
                            status, output = subprocess.getstatusoutput(self.current_invocation)
                            self.results_queue.put((self.current_invocation, status, output))

                        else:
                            raise TypeError(f"Unrecognized self.current_invocation {self.current_invocation}")
                    except KeyboardInterrupt:
                        break
                    except queue.Empty:
                        continue
                    except Exception as ex:
                        print("\n\n------------------")
                        print("[Error in process {}]: Ex = {}".format(self.ident, ex))
                        print("Current invocation:")
                        for e in self.current_invocation:
                            try:
                                pprint({k: v for k, v in e.items() if k != "input"})
                            except AttributeError:
                                pprint(e)
                        print(f"{type(ex).__name__}: {ex}")
                        print("==================")
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback_message = ""
                        for l in traceback.format_tb(exc_traceback):
                            traceback_message += l
                        print(traceback_message)
                        print("------------------\n\n")
                        self.results_queue.put((ex, self.current_invocation))
            finally:
                if self.verbose:
                    print(f"Exiting worker {self.id}")
                open(pid_file_path, "w").write(f"Exited: {self.current_invocation}...\n")


if __name__ == '__main__':
    import time


    def f(x):
        x = x ** 3 - 12 * x
        m = 27 * 29 * 33 * 37
        for _ in range(300000):
            x = (x ** 2 - 2 * x - 7) % m
        return x


    for process_count in range(1, multiprocessing.cpu_count() + 1):
        time_before = time.time()


        def on_function_call(function, kwargs, result):
            assert function == f
            assert list(kwargs.keys()) == ["x"]
            assert type(result) == int
            # print(f"{function.__name__}(**{kwargs}) = {result}")


        args = [(f, dict(x=x)) for x in range(100)]
        invoker = ParallelInvoker(invocation_list=args, process_count=process_count)
        results = invoker.run(on_result_func=on_function_call)
        assert len(results) == len(args)


        def print_shell_invocation(invocation, status, output):
            assert type(invocation) == str
            assert type(status) == int
            assert type(output) == str
            # print(f"shell('{invocation}') --[status={status}]--> {{\n{output}\n}}")
            pass


        args = [f"seq 1 {m} | grep {m % 4}" for m in range(1, 100)]
        invoker = ParallelInvoker(invocation_list=args, process_count=process_count)
        results = invoker.run(on_result_func=print_shell_invocation)
        assert len(results) == len(args)

        print(f"Time with {process_count} processes : {time.time() - time_before}")
