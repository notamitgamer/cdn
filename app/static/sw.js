// This service worker intentionally does NOT cache anything.
// It exists only so the site satisfies PWA "installability" requirements
// (Add to Home Screen / desktop install) while every request still goes
// straight to the network. Directory listings and files here can change
// at any time, so nothing is ever served from a cache.

self.addEventListener("install", (event) => {
  // Activate the new worker immediately, don't wait for old tabs to close.
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      // Defensive cleanup: if a cache was ever created by a previous
      // version of this worker, delete it so nothing stale lingers.
      const keys = await caches.keys();
      await Promise.all(keys.map((key) => caches.delete(key)));
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  // Always hit the network. No cache.match, no cache.put — ever.
  event.respondWith(fetch(event.request));
});
