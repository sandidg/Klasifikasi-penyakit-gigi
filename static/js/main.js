/**
 * ═══════════════════════════════════════════════════════════
 *  Deteksi Penyakit Gigi — Frontend Logic
 * ═══════════════════════════════════════════════════════════
 *  Handles:
 *    • File upload preview
 *    • Camera activation / capture
 *    • AJAX prediction request via Fetch API
 *    • Dynamic result rendering
 * ═══════════════════════════════════════════════════════════
 */

(function () {
    "use strict";

    // ── DOM Elements ──────────────────────────────────────
    const fileInput       = document.getElementById("file-input");
    const btnUploadLabel  = document.getElementById("btn-upload-label");
    const btnCamera       = document.getElementById("btn-camera");
    const btnCapture      = document.getElementById("btn-capture");
    const btnStopCamera   = document.getElementById("btn-stop-camera");
    const btnAnalyze      = document.getElementById("btn-analyze");

    const viewportEmpty   = document.getElementById("viewport-empty");
    const viewportImage   = document.getElementById("viewport-image");
    const viewportVideo   = document.getElementById("viewport-video");
    const viewportLoading = document.getElementById("viewport-loading");
    const captureCanvas   = document.getElementById("capture-canvas");

    const viewportStatus  = document.getElementById("viewport-status");
    const statusDot       = viewportStatus.querySelector(".status-dot");
    const statusText      = viewportStatus.querySelector(".status-text");

    const resultsPlaceholder = document.getElementById("results-placeholder");
    const resultsContent     = document.getElementById("results-content");

    // ── State ─────────────────────────────────────────────
    let cameraStream = null;
    let currentImageFile = null;      // File object
    let currentImageBase64 = null;    // base64 string (from camera)
    let isAnalyzing = false;


    // ═══════════════════════════════════════════════════════
    //  UTILITY FUNCTIONS
    // ═══════════════════════════════════════════════════════

    function showElement(el)  { el.classList.remove("d-none"); }
    function hideElement(el)  { el.classList.add("d-none"); }

    function setStatus(type, text) {
        statusDot.className = "status-dot";
        if (type) statusDot.classList.add(type);
        statusText.textContent = text;
    }

    function resetViewport() {
        hideElement(viewportImage);
        hideElement(viewportVideo);
        hideElement(viewportLoading);
        showElement(viewportEmpty);
        setStatus("", "Menunggu Input");
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach((t) => t.stop());
            cameraStream = null;
        }
        hideElement(viewportVideo);
        hideElement(btnCapture);
        hideElement(btnStopCamera);
        showElement(btnCamera);
    }


    // ═══════════════════════════════════════════════════════
    //  FILE UPLOAD
    // ═══════════════════════════════════════════════════════

    fileInput.addEventListener("change", function () {
        const file = this.files[0];
        if (!file) return;

        // Stop camera if running
        stopCamera();

        // Clear previous base64
        currentImageBase64 = null;
        currentImageFile = file;

        // Preview
        const reader = new FileReader();
        reader.onload = function (e) {
            viewportImage.src = e.target.result;
            hideElement(viewportEmpty);
            hideElement(viewportLoading);
            showElement(viewportImage);
            showElement(btnAnalyze);
            setStatus("active", "Gambar Siap");
        };
        reader.readAsDataURL(file);
    });


    // ═══════════════════════════════════════════════════════
    //  CAMERA
    // ═══════════════════════════════════════════════════════

    btnCamera.addEventListener("click", async function () {
        try {
            // Request camera — prefer rear camera on mobile
            const constraints = {
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);

            viewportVideo.srcObject = cameraStream;
            hideElement(viewportEmpty);
            hideElement(viewportImage);
            showElement(viewportVideo);

            // Show capture + stop buttons, hide camera button
            showElement(btnCapture);
            showElement(btnStopCamera);
            hideElement(btnCamera);
            hideElement(btnAnalyze);

            currentImageFile = null;
            currentImageBase64 = null;

            setStatus("active", "Kamera Aktif");
        } catch (err) {
            console.error("Camera error:", err);
            alert(
                "Tidak dapat mengakses kamera.\n\n" +
                "Pastikan:\n" +
                "• Browser memiliki izin kamera\n" +
                "• Halaman dibuka melalui HTTPS atau localhost"
            );
        }
    });

    btnCapture.addEventListener("click", function () {
        if (!cameraStream) return;

        const video = viewportVideo;
        const canvas = captureCanvas;

        // Match canvas to video dimensions
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        // Flip horizontally because video is mirrored via CSS
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert to base64
        currentImageBase64 = canvas.toDataURL("image/jpeg", 0.92);
        currentImageFile = null;

        // Stop camera and show captured image
        stopCamera();

        viewportImage.src = currentImageBase64;
        hideElement(viewportEmpty);
        showElement(viewportImage);
        showElement(btnAnalyze);
        setStatus("active", "Foto Berhasil Diambil");
    });

    btnStopCamera.addEventListener("click", function () {
        stopCamera();
        if (!viewportImage.src || viewportImage.classList.contains("d-none")) {
            resetViewport();
            hideElement(btnAnalyze);
        }
    });


    // ═══════════════════════════════════════════════════════
    //  ANALYZE (AJAX → /predict)
    // ═══════════════════════════════════════════════════════

    btnAnalyze.addEventListener("click", async function () {
        if (isAnalyzing) return;
        if (!currentImageFile && !currentImageBase64) {
            alert("Silakan unggah gambar atau ambil foto terlebih dahulu.");
            return;
        }

        isAnalyzing = true;
        showElement(viewportLoading);
        setStatus("processing", "Menganalisis…");
        hideElement(btnAnalyze);

        try {
            let response;

            if (currentImageFile) {
                // ── Send as multipart file ────────────────
                const formData = new FormData();
                formData.append("image", currentImageFile);
                response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
            } else if (currentImageBase64) {
                // ── Send as JSON base64 ───────────────────
                response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image_base64: currentImageBase64 })
                });
            }

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Terjadi kesalahan pada server.");
            }

            // ── Update viewport with result image ─────────
            // Add cache-buster
            viewportImage.src = data.result_image + "?t=" + Date.now();
            hideElement(viewportLoading);
            showElement(viewportImage);
            setStatus("done", "Deteksi Selesai");
            showElement(btnAnalyze);

            // ── Render results ────────────────────────────
            renderResults(data);

        } catch (err) {
            console.error("Prediction error:", err);
            hideElement(viewportLoading);
            showElement(viewportImage);
            setStatus("", "Gagal Menganalisis");
            showElement(btnAnalyze);
            alert("Gagal menganalisis gambar:\n" + err.message);
        } finally {
            isAnalyzing = false;
        }
    });


    // ═══════════════════════════════════════════════════════
    //  RENDER RESULTS
    // ═══════════════════════════════════════════════════════

    function renderResults(data) {
        hideElement(resultsPlaceholder);
        resultsContent.innerHTML = "";

        if (!data.detections || data.detections.length === 0) {
            resultsContent.innerHTML = `
                <div class="result-item">
                    <div class="result-header">
                        <div class="result-icon success">
                            <i class="bi bi-check-circle-fill"></i>
                        </div>
                        <div>
                            <div class="result-disease-name">Tidak Terdeteksi Penyakit</div>
                            <div class="result-badge" style="background:rgba(15,217,184,0.2);color:#4dead5;">
                                <i class="bi bi-check2"></i> HEALTHY
                            </div>
                        </div>
                    </div>
                    <div class="result-description">
                        Tidak ditemukan tanda-tanda penyakit gigi pada gambar yang diberikan.
                        Tetap jaga kesehatan gigi Anda dengan menyikat gigi 2× sehari dan
                        rutin periksa ke dokter gigi setiap 6 bulan.
                    </div>
                </div>`;
            showElement(resultsContent);
            return;
        }

        if (data.detections && data.detections.length > 0) {
            const det = data.detections[0];
            const confidence = Math.round(det.disease_confidence * 100);
            const isHighRisk = confidence >= 70;
            const severityClass = isHighRisk ? "danger" : "warning";
            const severityLabel = isHighRisk ? "RISIKO TINGGI" : "PERLU PERHATIAN";
            const iconClass = isHighRisk
                ? "bi-exclamation-triangle-fill"
                : "bi-info-circle-fill";

            const item = document.createElement("div");
            item.className = "result-item";
            item.style.animationDelay = "0s";

            let probHtml = "";
            if (det.all_probabilities) {
                probHtml += `
                <div class="result-all-probabilities mt-3 pt-3 border-top border-secondary-subtle" style="border-top: 1px dashed rgba(255,255,255,0.1) !important;">
                    <div class="mb-2 text-white-50" style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                        <i class="bi bi-bar-chart-fill me-1"></i> Probabilitas Dataset
                    </div>
                    <div class="d-flex flex-column gap-2">
                `;
                
                // Sort descending based on probability value
                const sortedProbs = Object.entries(det.all_probabilities).sort((a, b) => b[1] - a[1]);
                
                sortedProbs.forEach(([rawName, val]) => {
                    const pct = Math.round(val * 100);
                    let cleanName = rawName;
                    if (rawName.includes("Caries")) cleanName = "Caries";
                    else if (rawName.includes("Calculus")) cleanName = "Calculus";
                    else if (rawName.includes("Gingivitis")) cleanName = "Gingivitis";
                    else if (rawName.includes("Discoloration")) cleanName = "Tooth Discoloration";

                    probHtml += `
                        <div class="prob-row">
                            <div class="d-flex justify-content-between align-items-center mb-1" style="font-size: 0.75rem;">
                                <span class="text-light-emphasis">${escapeHtml(cleanName)}</span>
                                <span class="fw-semibold" style="color: var(--clr-primary-300);">${pct}%</span>
                            </div>
                            <div class="confidence-bar-bg" style="height: 4px; background: rgba(255,255,255,0.04); margin-bottom: 0;">
                                <div class="confidence-bar-fill" style="width: 0%; height: 100%; background: linear-gradient(90deg, var(--clr-primary-500), var(--clr-accent));" data-target="${pct}"></div>
                            </div>
                        </div>
                    `;
                });
                
                probHtml += `
                    </div>
                </div>
                `;
            }

            item.innerHTML = `
                <div class="result-header">
                    <div class="result-icon ${severityClass}">
                        <i class="bi ${iconClass}"></i>
                    </div>
                    <div>
                        <div class="result-disease-name">${escapeHtml(det.disease)}</div>
                        <div class="result-badge ${severityClass}">
                            <i class="bi bi-${isHighRisk ? "exclamation-diamond" : "dash-circle"}"></i>
                            ${severityLabel}
                        </div>
                    </div>
                </div>

                <div class="result-confidence">
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: 0%;" data-target="${confidence}"></div>
                    </div>
                    <span class="confidence-value">${confidence}%</span>
                </div>

                <div class="result-description">
                    ${escapeHtml(det.description)}
                </div>

                ${probHtml}

                <div class="result-timestamp">
                    <i class="bi bi-clock"></i>
                    ${formatTimestamp(data.timestamp)}
                </div>
            `;

            resultsContent.appendChild(item);
        }

        showElement(resultsContent);

        // Animate confidence bars after a tiny delay
        requestAnimationFrame(function () {
            setTimeout(function () {
                document.querySelectorAll(".confidence-bar-fill").forEach(function (bar) {
                    bar.style.width = bar.dataset.target + "%";
                });
            }, 100);
        });

        // Scroll results into view on mobile
        if (window.innerWidth < 992) {
            resultsContent.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }


    // ═══════════════════════════════════════════════════════
    //  HELPERS
    // ═══════════════════════════════════════════════════════

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    function formatTimestamp(isoString) {
        try {
            const d = new Date(isoString);
            return d.toLocaleString("id-ID", {
                day: "2-digit",
                month: "long",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"
            });
        } catch {
            return isoString;
        }
    }

})();
