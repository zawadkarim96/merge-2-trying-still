const statusEl = document.getElementById("status");
const saveBtn = document.getElementById("save-btn");
const pdfBtn = document.getElementById("pdf-btn");

const params = new URLSearchParams(window.location.search);
const quotationId = params.get("quotation_id");

const fields = {
  date: document.getElementById("date"),
  reference: document.getElementById("reference"),
  customer_company: document.getElementById("customer_company"),
  attention: document.getElementById("attention"),
  subject: document.getElementById("subject"),
  address: document.getElementById("address"),
  salutation: document.getElementById("salutation"),
  introduction: document.getElementById("introduction"),
  closing: document.getElementById("closing"),
};

function setStatus(message, tone = "info") {
  if (!statusEl) return;
  const palette = {
    info: "#cbd5e1",
    success: "#bbf7d0",
    error: "#fecdd3",
  };
  statusEl.textContent = message;
  statusEl.style.color = palette[tone] || palette.info;
}

function collectPayload() {
  return {
    date: fields.date.value || null,
    reference: fields.reference.value.trim(),
    customer_company: fields.customer_company.value.trim(),
    attention: fields.attention.value.trim(),
    subject: fields.subject.value.trim(),
    address: fields.address.value.trim(),
    salutation: fields.salutation.value.trim(),
    introduction: fields.introduction.value.trim(),
    closing: fields.closing.value.trim(),
    customer_contact: fields.attention.value.trim() || fields.customer_company.value.trim(),
  };
}

async function fetchQuotation() {
  if (!quotationId) {
    setStatus("Missing quotation_id in the URL", "error");
    return;
  }
  try {
    const resp = await fetch(`/api/quotation/${quotationId}`);
    if (!resp.ok) {
      throw new Error(`Unable to load quotation (${resp.status})`);
    }
    const data = await resp.json();
    fields.date.value = data.date || "";
    fields.reference.value = data.reference || "";
    fields.customer_company.value = data.customer_company || "";
    fields.attention.value = data.attention || "";
    fields.subject.value = data.subject || "";
    fields.address.value = data.address || "";
    fields.salutation.value = data.salutation || "";
    fields.introduction.value = data.introduction || "";
    fields.closing.value = data.closing || "";
    setStatus("Ready. Update the fields and save when done.", "success");
  } catch (err) {
    console.error(err);
    setStatus(err.message || "Failed to load quotation", "error");
  }
}

async function saveQuotation() {
  if (!quotationId) {
    setStatus("Missing quotation_id in the URL", "error");
    return;
  }
  setStatus("Saving...", "info");
  try {
    const resp = await fetch(`/api/quotation/${quotationId}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({}));
      throw new Error(payload.error || "Save failed");
    }
    setStatus("Quotation saved to CRM.", "success");
  } catch (err) {
    console.error(err);
    setStatus(err.message || "Unable to save", "error");
  }
}

async function downloadPdf() {
  if (!quotationId) {
    setStatus("Missing quotation_id in the URL", "error");
    return;
  }
  setStatus("Preparing PDF...", "info");
  try {
    const resp = await fetch(`/api/quotation/${quotationId}/pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({}));
      throw new Error(payload.error || "PDF generation failed");
    }
    const blob = await resp.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `quotation_${quotationId}.pdf`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    setStatus("PDF downloaded.", "success");
  } catch (err) {
    console.error(err);
    setStatus(err.message || "Unable to download PDF", "error");
  }
}

saveBtn?.addEventListener("click", saveQuotation);
pdfBtn?.addEventListener("click", downloadPdf);

window.addEventListener("load", fetchQuotation);
