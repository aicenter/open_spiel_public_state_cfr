#include <iostream>
#include <random>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "omp.h"

#include "spiel.h"

// Simplified implementation of fork-join parallelization of tasks with random
// duration, via OpenMP. Tasks in threads prepare inputs for a neural network.
// Once either a) entire batch is collected or b) timeout occurs, the batch is
// sent for inference on the GPU. The results are then provided for each thread
// for reading, and the loop continues.

#define TIMEOUT_us 1024  // In microseconds.
#define TIME_DIFF_us(start) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count()

struct Point {
  int data;   // input
  int target; // output
};

struct Batch {
  std::vector<Point> points;

  std::atomic<int> write_count = 0;  // Value in [0, NUM_THREADS]
  std::atomic<int> read_count = 0;   // Value in [0, write_count]
  std::vector<std::mutex> write_locks;
  // Semaphore for notifying that inference is done.
  std::mutex inference_mtx;
  std::condition_variable inference_done;
  bool ready = false;

  std::atomic<bool> make_inference = true;

  Batch(int size) : points(size), write_locks(size) {}

  Point* WriteablePoint(int thread_idx) {
    return &points[thread_idx];
  }
  void NotifyWritten() { ++write_count; }

  const Point& ReadablePoint(int thread_idx) const {
    return points[thread_idx];
  }
  void NotifyRead() { ++read_count; }

  int size() const { return points.size(); }
  bool IsFilled() const { return write_count == size(); }

  void Reset() {
    ready = false;
    write_count = 0;
    read_count = 0;
    for (Point& point : points) {
      point.data = 0;
      point.target = 0;
    }
  }
};

// Emulates neural network: just multiply everything by 2.
void RunInference(Batch& batch) {
  for (Point& point : batch.points) {
    point.target = point.data * 2;
  }
}

// Run inference continually in a separate thread.
[[noreturn]] void ContinualInference(Batch* batch) {
  const auto sleep_step =
      std::chrono::microseconds(TIMEOUT_us / (8 * batch->size()));

  while (batch->make_inference) {
    // Wait until we get a first write.
    while (batch->write_count == 0) std::this_thread::sleep_for(sleep_step);
    // We got first write(s), let's set the entry point time.
    auto entry_point = std::chrono::system_clock::now();
    // Wait until batch is filled or timeout.
    while (!batch->IsFilled() && TIME_DIFF_us(entry_point) < TIMEOUT_us)
      std::this_thread::sleep_for(sleep_step);

    // Wait until all writes are finished.
    for (int i = 0; i < batch->size(); ++i) batch->write_locks[i].lock();

    {
      std::lock_guard<std::mutex> lck{batch->inference_mtx};
      RunInference(*batch);
      // Notify the waiting threads about inference being done.
      batch->ready = true;
      batch->inference_done.notify_all();
    }

    // Wait until all reads are finished.
    while (batch->read_count != batch->write_count) {
      std::this_thread::sleep_for(sleep_step);
    }
    // Everything has been read, we can safely reset batch.
    batch->Reset();

    // And now threads can write next.
    for (int i = 0; i < batch->size(); ++i) batch->write_locks[i].unlock();
  }
}

int main(int argc, char* argv[]) {
  int num_threads = std::stoi(argv[1]);
  std::cout << "Running with " << num_threads << "\n";
  Batch batch(num_threads);
  std::thread inference_thread(ContinualInference, &batch);

#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < 50000; i++) {
    // Inputs and outputs for current iteration with random length.
    std::mt19937 rnd_gen(i);
    int random_number_of_jobs = std::uniform_int_distribution<>(0, 16)(rnd_gen);
    int thread_id = omp_get_thread_num();
//    std::cout << "Start loop #" << i << " (" << random_number_of_jobs
//              << " jobs) using thread " << thread_id << std::endl;
    for (int j = 1; j <= random_number_of_jobs; ++j) {
      // Make up some input data, this can take a while (random sleep).
      int x = std::uniform_int_distribution<>(0, 100000)(rnd_gen);

      const int t = std::uniform_int_distribution<>(1000, 10000)(rnd_gen);
      int random_workload = 1;
      for (int k = t; k <= 10000; ++k) { random_workload *= k; }
      if (random_workload < 0) std::cout << '.';

      {
        batch.write_locks[thread_id].lock();

        Point* point = batch.WriteablePoint(thread_id);
        point->data = x;
        batch.NotifyWritten();

        batch.write_locks[thread_id].unlock();
      }

      // Inference runs in separate thread! See ContinualInference()
      std::unique_lock<std::mutex> lck(batch.inference_mtx);
      batch.inference_done.wait(lck, [&]() { return batch.ready; });

      {
        // Make sure we got the right answer.
        const Point& point = batch.ReadablePoint(thread_id);
        if (point.target != x * 2) open_spiel::SpielFatalError("Wrong target");
        batch.NotifyRead();
      }

//      std::cout << "Finished #" << i
//                << " (" << j << '/' << random_number_of_jobs
//                << ") using thread " << thread_id << std::endl;
    }
  }

  batch.make_inference = false;
  inference_thread.join();

  std::cout << "All done! Hurray!" << std::endl;
}
