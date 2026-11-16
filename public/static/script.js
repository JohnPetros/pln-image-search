const form = document.getElementById("search-form");
const input = document.getElementById("search-input");
const result = document.getElementById("result");
const answer = document.getElementById("answer");
const debug = document.getElementById("debug");
const gallery = document.getElementById("gallery");

const CATEGORY_MODIFIER = {
  "vegetação": "vegetacao",
  "solo exposto": "solo-exposto",
  "água": "agua",
};

form.addEventListener("submit", (event) => {
  event.preventDefault();
  search(input.value);
});

document.getElementById("examples").addEventListener("click", (event) => {
  const chip = event.target.closest(".examples__chip");
  if (!chip) return;
  input.value = chip.textContent.trim();
  search(input.value);
});

async function search(query) {
  show("Buscando…", "", []);
  try {
    const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error(response.statusText);
    const data = await response.json();
    show(data.answer, describePipeline(data), data.images);
  } catch (error) {
    show("Não foi possível consultar o servidor.", String(error), []);
  }
}

function describePipeline(data) {
  if (!data.intent) return "";
  const params = Object.entries(data.params)
    .filter(([, value]) => value !== null)
    .map(([key, value]) => `${key}=${value}`);
  const parts = [`intenção: ${data.intent}`, `cosseno: ${data.score.toFixed(2)}`];
  if (params.length) parts.push(`parâmetros: ${params.join(", ")}`);
  return parts.join(" · ");
}

function show(answerText, debugText, images) {
  result.hidden = false;
  answer.textContent = answerText;
  debug.textContent = debugText;
  gallery.replaceChildren(...images.map(buildCard));
}

function buildCard(image) {
  const card = document.createElement("figure");
  card.className = "card";

  const img = document.createElement("img");
  img.className = "card__image";
  img.src = `/${image.arquivo}`;
  img.alt = `Imagem de ${image.categoria} capturada por ${image.origem}`;
  img.loading = "lazy";

  const caption = document.createElement("figcaption");
  caption.className = "card__caption";

  const badge = document.createElement("span");
  badge.className = `card__badge card__badge--${CATEGORY_MODIFIER[image.categoria]}`;
  badge.textContent = image.categoria;

  const meta = document.createElement("span");
  meta.className = "card__meta";
  meta.textContent = `${image.origem} · ${image.data}`;

  caption.append(badge, meta);
  card.append(img, caption);
  return card;
}
