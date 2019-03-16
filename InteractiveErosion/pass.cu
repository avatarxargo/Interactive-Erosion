#include "pass.cuh"

namespace pass {
	SimMap* map;
	HANDLE handle;
	std::mutex m;
	std::condition_variable cv;
	unsigned int requested = 0;
	bool active = true;
	volatile bool activityRequest = false;

	void activity() {
		//lock simMap
		DWORD dwWaitResult;
		dwWaitResult = WaitForSingleObject(
			handle,    // handle to mutex
			INFINITE);  // no time-out interval
		map->passLayerListPointers();
		map->passMaterialListPointers();
		cudaError_t err = cudaGetLastError();
		if (err != 0) {
			printf("XXX Pass CUDA encountered an error number %d! \n", err);
		}
		//unlock simMap
		ReleaseMutex(handle);
	}

	void worker_thread()
	{
		while (true) {
			// Wait until main() sends data
			std::unique_lock<std::mutex> lk(m);
			cv.wait(lk, [] {return activityRequest; });
			if (!active) { lk.unlock(); return; }

			// after the wait, we own the lock.
			activity();

			// Manual unlocking is done before notifying, to avoid waking up
			// the waiting thread only to block again (see notify_one for details)
			lk.unlock();
			activityRequest = false;
		}
	}

	void killThread() {
		printf("Terminating Pass Thread\n");
		active = false;
		cv.notify_one();
	}

	void initThread() {
		printf("Initing Pass Thread\n");
		std::thread worker(worker_thread);
		worker.detach();
	}

	void passData(SimMap* simMap, HANDLE handl) {
		map = simMap;
		handle = handl;
		//ping thread
		activityRequest = true;
		cv.notify_one();
	}
}
