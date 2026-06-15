import { t, getLang, setLang } from "./i18n.js";

// ---------- State ----------
let MOVIES = [];      // sorted by popularity (desc) as produced by the engine
let GENRES = [];
let TITLE_INDEX = []; // { i, lc } for fast search
let lang = getLang();

const POSTER_FALLBACK =
  "data:image/svg+xml," +
  encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 300'>
      <rect width='200' height='300' fill='#1c1c25'/>
      <text x='100' y='150' fill='#4a4a55' font-size='20' font-family='Arial'
        text-anchor='middle'>No poster</text>
    </svg>`
  );

const app = document.getElementById("app");
const $ = (sel) => document.querySelector(sel);

// ---------- Helpers ----------
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

const poster = (m, cls) =>
  `<img class="${cls}" loading="lazy" alt="${esc(m.title)}"
     src="${m.poster ? esc(m.poster) : POSTER_FALLBACK}"
     onerror="this.onerror=null;this.src='${POSTER_FALLBACK}'" />`;

function cardHTML(i) {
  const m = MOVIES[i];
  const rating = m.rating ? `<span class="rating-badge">★ ${m.rating}</span>` : "";
  return `<a class="card" href="#/movie/${i}">
    ${poster(m, "poster")}${rating}
    <div class="info">
      <div class="t">${esc(m.title)}</div>
      <div class="y">${m.year ?? "—"}</div>
    </div>
  </a>`;
}

const rowHTML = (indices) =>
  `<div class="row">${indices.map(cardHTML).join("")}</div>`;
const gridHTML = (indices) =>
  `<div class="grid">${indices.map(cardHTML).join("")}</div>`;

function moviesInGenre(genre) {
  const out = [];
  for (let i = 0; i < MOVIES.length; i++)
    if (MOVIES[i].genres.includes(genre)) out.push(i);
  return out; // already in popularity order
}

const HOME_GENRES = [
  "Action",
  "Comedy",
  "Drama",
  "Animation",
  "Horror",
  "Science Fiction",
];

// ---------- Views ----------
function renderHome() {
  const popular = [...Array(20).keys()];
  const topRated = [...Array(Math.min(1000, MOVIES.length)).keys()]
    .sort((a, b) => MOVIES[b].rating - MOVIES[a].rating)
    .slice(0, 20);

  const chips = GENRES.map(
    (g) => `<a class="chip" href="#/genre/${encodeURIComponent(g)}">${esc(g)}</a>`
  ).join("");

  const genreSections = HOME_GENRES.filter((g) => GENRES.includes(g))
    .map((g) => {
      const items = moviesInGenre(g).slice(0, 18);
      if (!items.length) return "";
      return `<section class="section">
        <h2><a href="#/genre/${encodeURIComponent(g)}">${esc(g)} →</a></h2>
        ${rowHTML(items)}
      </section>`;
    })
    .join("");

  app.innerHTML = `
    <section class="hero">
      <h1>${esc(t("tagline_title", lang))}</h1>
      <p>${esc(t("tagline_sub", lang))}</p>
      <div class="chips">${chips}</div>
    </section>
    <div class="wrap">
      <section class="section"><h2>${esc(t("popular", lang))}</h2>${rowHTML(
    popular
  )}</section>
      <section class="section"><h2>${esc(t("top_rated", lang))}</h2>${rowHTML(
    topRated
  )}</section>
      ${genreSections}
    </div>`;
}

function renderMovie(i) {
  const m = MOVIES[i];
  if (!m) return renderHome();

  const recs = (m.rec || []).filter((j) => MOVIES[j]);
  const genres = m.genres
    .map(
      (g) =>
        `<a class="chip" href="#/genre/${encodeURIComponent(g)}">${esc(g)}</a>`
    )
    .join("");

  app.innerHTML = `<div class="wrap detail">
    <button class="back-btn" onclick="history.back()">← ${esc(
      t("back", lang)
    )}</button>
    <div class="detail-top">
      ${poster(m, "detail-poster")}
      <div class="detail-info">
        <h1>${esc(m.title)}</h1>
        <div class="meta-row">
          ${m.year ? `<span class="pill">${m.year}</span>` : ""}
          ${
            m.rating
              ? `<span class="star">★ ${m.rating}</span><span>/ 10</span>`
              : ""
          }
          ${
            m.lang
              ? `<span class="pill">${esc(m.lang.toUpperCase())}</span>`
              : ""
          }
        </div>
        <div class="detail-genres">${genres}</div>
        <h3>${esc(t("overview", lang))}</h3>
        <p class="overview">${esc(m.overview || t("no_overview", lang))}</p>
      </div>
    </div>
    <section class="section">
      <h2>${esc(t("more_like_this", lang))}</h2>
      ${recs.length ? gridHTML(recs) : `<p class="empty">${esc(t("no_results", lang))}</p>`}
    </section>
  </div>`;
  window.scrollTo(0, 0);
}

let genreShown = 60;
function renderGenre(genre, more = false) {
  if (!more) genreShown = 60;
  const all = moviesInGenre(genre);
  const shown = all.slice(0, genreShown);
  const moreBtn =
    all.length > genreShown
      ? `<button class="show-more" id="more">${esc(t("show_more", lang))}</button>`
      : "";

  app.innerHTML = `<div class="wrap">
    <div class="page-head">
      <button class="back-btn" onclick="location.hash='#/'">← ${esc(
        t("home", lang)
      )}</button>
      <div class="sub">${esc(t("genre", lang))}</div>
      <h1>${esc(genre)} <span class="sub">· ${all.length}</span></h1>
    </div>
    ${all.length ? gridHTML(shown) : `<p class="empty">${esc(t("no_results", lang))}</p>`}
    ${moreBtn}
  </div>`;
  const btn = $("#more");
  if (btn)
    btn.onclick = () => {
      genreShown += 60;
      renderGenre(genre, true);
    };
  window.scrollTo(0, 0);
}

function searchIndices(query, limit = Infinity) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const starts = [];
  const contains = [];
  for (const { i, lc } of TITLE_INDEX) {
    const pos = lc.indexOf(q);
    if (pos === 0) starts.push(i);
    else if (pos > 0) contains.push(i);
    if (starts.length >= limit) break;
  }
  return starts.concat(contains).slice(0, limit); // popularity order preserved
}

function renderSearch(query) {
  const results = searchIndices(query);
  app.innerHTML = `<div class="wrap">
    <div class="page-head">
      <div class="sub">${esc(t("results_for", lang))}</div>
      <h1>“${esc(query)}” <span class="sub">· ${results.length}</span></h1>
    </div>
    ${results.length ? gridHTML(results) : `<p class="empty">${esc(t("no_results", lang))}</p>`}
  </div>`;
  window.scrollTo(0, 0);
}

// ---------- Router ----------
function router() {
  const hash = location.hash.slice(1); // e.g. /movie/12
  const [, route, param] = hash.split("/");
  if (route === "movie") renderMovie(Number(param));
  else if (route === "genre") renderGenre(decodeURIComponent(param || ""));
  else if (route === "search") renderSearch(decodeURIComponent(param || ""));
  else renderHome();
}

// ---------- Search box (suggestions) ----------
function setupSearch() {
  const input = $("#search");
  const box = $("#suggestions");
  let active = -1;
  let current = [];

  const close = () => {
    box.hidden = true;
    active = -1;
  };

  const open = (q) => {
    current = searchIndices(q, 8);
    if (!current.length) return close();
    box.innerHTML = current
      .map(
        (i, n) => `<li data-i="${i}" class="${n === active ? "active" : ""}">
          ${poster(MOVIES[i], "")}
          <div>
            <div class="s-title">${esc(MOVIES[i].title)}</div>
            <div class="s-meta">${MOVIES[i].year ?? "—"} · ★ ${
          MOVIES[i].rating || "–"
        }</div>
          </div>
        </li>`
      )
      .join("");
    box.hidden = false;
    box.querySelectorAll("li").forEach((li) => {
      li.onclick = () => {
        location.hash = `#/movie/${li.dataset.i}`;
        input.value = "";
        close();
      };
    });
  };

  let timer;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value;
    timer = setTimeout(() => open(q), 110);
  });

  input.addEventListener("keydown", (e) => {
    if (box.hidden) {
      if (e.key === "Enter" && input.value.trim())
        location.hash = `#/search/${encodeURIComponent(input.value.trim())}`;
      return;
    }
    const items = [...box.querySelectorAll("li")];
    if (e.key === "ArrowDown") {
      e.preventDefault();
      active = (active + 1) % items.length;
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      active = (active - 1 + items.length) % items.length;
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (active >= 0) location.hash = `#/movie/${items[active].dataset.i}`;
      else location.hash = `#/search/${encodeURIComponent(input.value.trim())}`;
      input.value = "";
      return close();
    } else if (e.key === "Escape") {
      return close();
    } else {
      return;
    }
    items.forEach((li, n) => li.classList.toggle("active", n === active));
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-wrap")) close();
  });
}

// ---------- Language ----------
function applyChrome() {
  document.documentElement.lang = lang;
  $("#search").placeholder = t("search_placeholder", lang);
  $("#footer-data").textContent = t("footer_data", lang);
  $("#footer-note").textContent = t("footer_note", lang);
  document
    .querySelectorAll(".lang-toggle button")
    .forEach((b) => b.classList.toggle("active", b.dataset.lang === lang));
}

function setupLang() {
  document.querySelectorAll(".lang-toggle button").forEach((b) => {
    b.onclick = () => {
      lang = b.dataset.lang;
      setLang(lang);
      applyChrome();
      router(); // re-render current view in the new language
    };
  });
}

// ---------- Boot ----------
async function boot() {
  $("#splash-text").textContent = t("loading", lang);
  try {
    const res = await fetch("data/movies.json");
    const data = await res.json();
    MOVIES = data.movies;
    GENRES = data.genres;
    TITLE_INDEX = MOVIES.map((m, i) => ({ i, lc: m.title.toLowerCase() }));
  } catch (err) {
    $("#splash").innerHTML = `<p style="color:#e50914">Failed to load movie data.</p>`;
    console.error(err);
    return;
  }

  applyChrome();
  setupSearch();
  setupLang();
  window.addEventListener("hashchange", router);
  router();

  const splash = $("#splash");
  splash.classList.add("hide");
  setTimeout(() => splash.remove(), 450);
}

boot();
