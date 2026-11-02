/* Taloona Trade app: notifications from the bots' manager (Web Push) and the "approve update" button. */
const STATE_URL = "https://raw.githubusercontent.com/talshahino/trading-system/live/live/manager_state.json";
const SHELL = "taloona-v3";

self.addEventListener("install", e => { self.skipWaiting(); e.waitUntil(caches.open(SHELL).then(c => c.addAll(["./", "./index.html", "./icon-192.png"])).catch(() => {})); });
self.addEventListener("activate", e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== SHELL).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin) return;     // live data always from the network
  e.respondWith(fetch(e.request, {cache: "no-cache"}).then(r => { const c = r.clone(); caches.open(SHELL).then(x => x.put(e.request, c)); return r; })
    .catch(() => caches.match(e.request).then(r => r || caches.match("./index.html"))));
});

/* tiny key-value store shared with the page (the secret code and the manager address) */
function kv(key, val) {
  return new Promise((res, rej) => {
    const o = indexedDB.open("ftmo-bots", 1);
    o.onupgradeneeded = () => o.result.createObjectStore("kv");
    o.onerror = () => rej(o.error);
    o.onsuccess = () => {
      const tx = o.result.transaction("kv", val === undefined ? "readonly" : "readwrite"), st = tx.objectStore("kv");
      const q = val === undefined ? st.get(key) : st.put(val, key);
      q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error);
    };
  });
}

self.addEventListener("push", e => {
  let d = {};
  try { d = e.data.json(); } catch (x) { d = {title: "Taloona Trade", body: e.data ? e.data.text() : ""}; }
  const opt = {body: d.body || "", dir: "rtl", lang: "he", icon: "icon-192.png", badge: "badge.png",
               silent: !!d.silent, tag: d.tag || undefined, renotify: !!d.tag, data: d, requireInteraction: d.action === "approve_update"};
  if (d.action === "approve_update") opt.actions = [{action: "approve", title: "אשר עדכון"}, {action: "later", title: "אחר כך"}];
  e.waitUntil(self.registration.showNotification(d.title || "Taloona Trade", opt));
});

async function managerUrl() {
  const saved = await kv("mgr");                 // the address the app last connected to (newest)
  if (saved) { try { const r = await fetch(saved + "/status", {cache: "no-store"}); if (r.ok) return saved; } catch (x) {} }
  try { const r = await fetch(STATE_URL + "?t=" + Date.now(), {cache: "no-store"}); if (r.ok) { const j = await r.json(); if (j.url) return j.url; } } catch (x) {}
  return saved;
}

self.addEventListener("notificationclick", e => {
  e.notification.close();
  if (e.action === "later") return;
  if (e.action === "approve") {
    e.waitUntil((async () => {
      const pin = await kv("pin"), url = await managerUrl();
      let msg = "אין קוד סודי שמור באפליקציה. פתח את האפליקציה ואשר שם.";
      if (pin && url) {
        try {
          const r = await fetch(url + "/update/approve", {method: "POST", headers: {"Content-Type": "application/json", "X-Pin": pin}, body: "{}"});
          msg = r.ok ? "העדכון אושר. הבוטים מתעדכנים ועולים מחדש תוך כדקה." : "האישור נכשל: " + ((await r.json()).error || r.status);
        } catch (x) { msg = "אין חיבור למחשב הבוטים. נסה שוב מתוך האפליקציה."; }
      }
      await self.registration.showNotification("Taloona Trade", {body: msg, dir: "rtl", lang: "he", icon: "icon-192.png", badge: "badge.png", tag: "update"});
    })());
    return;
  }
  e.waitUntil(self.clients.matchAll({type: "window", includeUncontrolled: true}).then(cs => {
    for (const c of cs) if ("focus" in c) return c.focus();
    return self.clients.openWindow("./");
  }));
});
