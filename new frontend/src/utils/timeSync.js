// Singleton for time synchronization across components
class TimeSync {
  constructor() {
    this.listeners = new Set();
    this.startTime = Date.now();
    this.currentTime = 0;
    this.intervalId = null;
    this.start(); // Start immediately on construction
  }

  start() {
    if (this.intervalId) this.stop(); // Clear any existing interval

    this.startTime = Date.now();
    this.intervalId = setInterval(() => {
      this.currentTime = Math.floor((Date.now() - this.startTime) / 1000);
      this.notifyListeners();
    }, 1000);
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  reset() {
    this.stop();
    this.startTime = Date.now();
    this.currentTime = 0;
    this.notifyListeners();
    this.start();
  }

  subscribe(listener) {
    this.listeners.add(listener);
    // Immediately notify the new listener of current time
    listener(this.currentTime);
    return () => {
      this.listeners.delete(listener);
    };
  }

  notifyListeners() {
    this.listeners.forEach((listener) => {
      listener(this.currentTime);
    });
  }
}

// Export a singleton instance
export const timeSync = new TimeSync();
