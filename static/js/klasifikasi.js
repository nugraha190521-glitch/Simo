const MAX_BYTES = 10 * 1024 * 1024;

// ── Upload & Preview
const fileInput = document.getElementById("fileInput");
const uploadArea = document.getElementById("uploadArea");
const previewWrap = document.getElementById("previewWrap");
const previewImg = document.getElementById("previewImg");
const previewMeta = document.getElementById("previewMeta");
const btnReset = document.getElementById("btnReset");
const btnClassify = document.getElementById("btnClassify");

// ── Result
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultWrap = document.getElementById("resultWrap");
const resultImg = document.getElementById("resultImg");
const resultImgLabel = document.getElementById("resultImgLabel");
const topLabel = document.getElementById("topLabel");
const topConf = document.getElementById("topConf");
const lowConfNotice = document.getElementById("lowConfNotice");
const notMolluscNotice = document.getElementById("notMolluscNotice");
const topPred = document.getElementById("topPred");
const probList = document.getElementById("probList");
const spinner = document.getElementById("spinner");

let currentFile = null;

function showPreview(file) {
 currentFile = file;
 previewImg.src = URL.createObjectURL(file);
 previewMeta.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB`;
 uploadArea.classList.add("d-none");
 previewWrap.classList.remove("d-none");
 btnClassify.disabled = false;
 resetResult();
}

function resetUpload() {
 currentFile = null;
 fileInput.value = "";
 previewImg.src = "";
 uploadArea.classList.remove("d-none");
 previewWrap.classList.add("d-none");
 btnClassify.disabled = true;
 resetResult();
}

function resetResult() {
 resultPlaceholder.classList.remove("d-none");
 spinner.classList.add("d-none");
 resultWrap.classList.add("d-none");
 lowConfNotice.classList.add("d-none");
 notMolluscNotice.classList.add("d-none");
 topPred.classList.remove("d-none");
 probList.classList.remove("d-none");
}

function acceptFile(file) {
 if (!file) return;
 if (file.size > MAX_BYTES) {
  alert("Ukuran file maksimal 10MB!");
  return;
 }
 if (!file.type.startsWith("image/")) {
  alert("File harus berupa gambar.");
  return;
 }
 showPreview(file);
}

fileInput.addEventListener("change", (e) => {
 acceptFile(e.target.files[0]);
 if (!currentFile) fileInput.value = "";
});

uploadArea.addEventListener("dragover", (e) => {
 e.preventDefault();
 uploadArea.classList.add("is-drag");
});
uploadArea.addEventListener("dragleave", () =>
 uploadArea.classList.remove("is-drag"),
);
uploadArea.addEventListener("drop", (e) => {
 e.preventDefault();
 uploadArea.classList.remove("is-drag");
 acceptFile(e.dataTransfer.files[0]);
});

btnReset.addEventListener("click", resetUpload);

function renderResult(data) {
 // Gambar hasil BiRefNet (yang benar-benar masuk ke model)
 resultImg.src = data.processed_image || previewImg.src;

 topLabel.textContent = data.label;
 topConf.textContent = (data.confidence * 100).toFixed(2) + "%";
 const bukan = data.is_mollusc === false;
 console.log("similarity:", data.similarity); // untuk tuning threshold, boleh dihapus nanti
 resultImgLabel.textContent = bukan ? "bukan moluska" : data.label;
 notMolluscNotice.classList.toggle("d-none", !bukan);
 topPred.classList.toggle("d-none", bukan); // sembunyikan label + persen yang menyesatkan
 probList.classList.toggle("d-none", bukan);
 lowConfNotice.classList.toggle("d-none", bukan || !data.low_confidence);

 // Probabilitas semua kelas
 probList.replaceChildren();
 data.all_probs.forEach((p, i) => {
  const pct = (p.prob * 100).toFixed(2);

  const item = document.createElement("div");
  item.className = "mb-3";

  const row = document.createElement("div");
  row.className = "prob-row d-flex justify-content-between mb-1";
  const name = document.createElement("span");
  name.className = "prob-name";
  name.textContent = p.label;
  const val = document.createElement("span");
  val.className = "prob-val";
  val.textContent = pct + "%";
  row.append(name, val);

  const bg = document.createElement("div");
  bg.className = "prob-bar-bg";
  const bar = document.createElement("div");
  bar.className = "prob-bar " + (i === 0 ? "top" : "rest");
  bar.dataset.w = pct + "%";
  bg.append(bar);

  item.append(row, bg);
  probList.append(item);
 });

 requestAnimationFrame(() => {
  probList.querySelectorAll(".prob-bar").forEach((b) => {
   b.style.width = b.dataset.w;
  });
 });

 spinner.classList.add("d-none");
 resultWrap.classList.remove("d-none");
}

// ── Classify
btnClassify.addEventListener("click", async () => {
 if (!currentFile) return;

 resultPlaceholder.classList.add("d-none");
 resultWrap.classList.add("d-none");
 spinner.classList.remove("d-none");
 btnClassify.disabled = true;

 try {
  const formData = new FormData();
  formData.append("file", currentFile);

  const res = await fetch("/predict", { method: "POST", body: formData });
  const data = await res.json();

  if (data.error) {
   alert("Error: " + data.error);
   resetResult();
   return;
  }
  renderResult(data);
 } catch (err) {
  alert("Gagal menghubungi server: " + err.message);
  resetResult();
 } finally {
  btnClassify.disabled = !currentFile;
  spinner.classList.add("d-none");
 }
});
