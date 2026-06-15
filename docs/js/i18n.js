// Bilingual UI strings (EN/TR). Movie data itself stays in English — this only
// covers the interface chrome. Selected language persists in localStorage.
export const STRINGS = {
  en: {
    tagline_title: "Find your next favorite film",
    tagline_sub:
      "Search 9,500+ titles and get instant, content-based recommendations.",
    search_placeholder: "Search for a movie…",
    popular: "Popular now",
    top_rated: "Top rated",
    browse_genres: "Browse by genre",
    more_like_this: "More like this",
    overview: "Overview",
    no_results: "No movies found.",
    results_for: "Results for",
    back: "Back",
    genre: "Genre",
    rating: "Rating",
    language: "Language",
    year: "Year",
    loading: "Loading movies…",
    show_more: "Show more",
    no_overview: "No overview available.",
    footer_data: "Movie data from TMDB · CC0 public domain.",
    footer_note:
      "Content-based recommendations (TF-IDF + cosine similarity). Not affiliated with Netflix or TMDB.",
    home: "Home",
  },
  tr: {
    tagline_title: "Bir sonraki favori filmini bul",
    tagline_sub:
      "9.500+ film içinde ara, anında içerik-tabanlı öneriler al.",
    search_placeholder: "Film ara…",
    popular: "Şu an popüler",
    top_rated: "En yüksek puanlı",
    browse_genres: "Türe göre keşfet",
    more_like_this: "Buna benzer filmler",
    overview: "Özet",
    no_results: "Film bulunamadı.",
    results_for: "Arama sonuçları",
    back: "Geri",
    genre: "Tür",
    rating: "Puan",
    language: "Dil",
    year: "Yıl",
    loading: "Filmler yükleniyor…",
    show_more: "Daha fazla göster",
    no_overview: "Özet bulunmuyor.",
    footer_data: "Film verisi TMDB'den · CC0 kamu malı.",
    footer_note:
      "İçerik-tabanlı öneriler (TF-IDF + kosinüs benzerliği). Netflix veya TMDB ile bağlantılı değildir.",
    home: "Anasayfa",
  },
};

const LANGS = ["en", "tr"];

export function getLang() {
  const saved = localStorage.getItem("lang");
  if (LANGS.includes(saved)) return saved;
  return (navigator.language || "en").startsWith("tr") ? "tr" : "en";
}

export function setLang(lang) {
  if (LANGS.includes(lang)) localStorage.setItem("lang", lang);
}

export function t(key, lang = getLang()) {
  return (STRINGS[lang] && STRINGS[lang][key]) || STRINGS.en[key] || key;
}
