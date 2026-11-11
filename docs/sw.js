/* Taloona Trade app: notifications from the bots' manager (Web Push) and the "approve update" button. */
const STATE_URL = "https://raw.githubusercontent.com/talshahino/trading-system/live/live/manager_state.json";
const SHELL = "taloona-v23";

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

const MGR_API = "https://api.github.com/repos/talshahino/trading-system/contents/live/manager_state.json?ref=live";

const hex2b = h => Uint8Array.from(h.match(/../g).map(x => parseInt(x, 16)));
const b64b = x => Uint8Array.from(atob(x), c => c.charCodeAt(0));
async function openUrl(j, pin) {
  if (!j) return null;
  if (j.url) return j.url;
  let uk = null;
  try { uk = await kv("url_key"); } catch (x) {}
  if (uk) for (const e of (j.url_enc || []).filter(x => x.k)) {
    try {
      const key = await crypto.subtle.importKey("raw", hex2b(uk), "AES-GCM", false, ["decrypt"]);
      return new TextDecoder().decode(await crypto.subtle.decrypt({name: "AES-GCM", iv: b64b(e.iv)}, key, b64b(e.ct)));
    } catch (x) {}
  }
  for (const e of (j.url_enc || []).filter(x => x.s)) {
    try {
      const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(pin || ""), "PBKDF2", false, ["deriveBits"]);
      const bits = new Uint8Array(await crypto.subtle.deriveBits({name: "PBKDF2", hash: "SHA-256", salt: hex2b(e.s), iterations: 200000}, base, 256));
      const kb = await crypto.subtle.digest("SHA-256", new Uint8Array([...bits, ...new TextEncoder().encode("url")]));
      const key = await crypto.subtle.importKey("raw", kb, "AES-GCM", false, ["decrypt"]);
      return new TextDecoder().decode(await crypto.subtle.decrypt({name: "AES-GCM", iv: b64b(e.iv)}, key, b64b(e.ct)));
    } catch (x) {}
  }
  return null;
}

async function alive(u) {
  if (!u) return false;
  try { const r = await fetch(u + "/status", {cache: "no-store"}); return r.ok; } catch (x) { return false; }
}

async function managerUrl() {
  // the computer's address changes whenever the manager restarts: try the one the app last used, then the
  // published one (the file server can hand out a copy a few minutes old), then GitHub's API (always fresh)
  const saved = await kv("mgr");
  if (await alive(saved)) return saved;
  let raw = null, api = null;
  const pin = await kv("pin");
  try { const r = await fetch(STATE_URL + "?t=" + Date.now(), {cache: "no-store"}); if (r.ok) raw = await openUrl(await r.json(), pin); } catch (x) {}
  if (raw && raw !== saved && await alive(raw)) { await kv("mgr", raw); return raw; }
  try {
    const r = await fetch(MGR_API + "&t=" + Date.now(), {cache: "no-store", headers: {Accept: "application/vnd.github.raw+json"}});
    if (r.ok) api = await openUrl(await r.json(), pin);
  } catch (x) {}
  if (api && await alive(api)) { await kv("mgr", api); return api; }
  return api || raw || saved;
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
          const sha = (e.notification.data && e.notification.data.sha) || "";
          const r = await fetch(url + "/update/approve", {method: "POST", headers: {"Content-Type": "application/json", "X-Pin": pin}, body: JSON.stringify({sha})});
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
