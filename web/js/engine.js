/* Glyph.ai local engine: preprocess, lexicon, Tesseract, compression. */
(function (global) {
  "use strict";

  var MAX_SIDE = 1600;
  var worker = null;
  var workerReady = null;

  function loadImage(src) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () { resolve(img); };
      img.onerror = function () { reject(new Error("Не удалось открыть изображение")); };
      img.src = src;
    });
  }

  function fileToUrl(file) {
    return URL.createObjectURL(file);
  }

  function drawScaled(img) {
    var w = img.naturalWidth || img.width;
    var h = img.naturalHeight || img.height;
    var scale = Math.min(1, MAX_SIDE / Math.max(w, h));
    var cw = Math.max(1, Math.round(w * scale));
    var ch = Math.max(1, Math.round(h * scale));
    var canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    var ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, cw, ch);
    return canvas;
  }

  function preprocessCanvas(srcCanvas) {
    var w = srcCanvas.width;
    var h = srcCanvas.height;
    var ctx = srcCanvas.getContext("2d", { willReadFrequently: true });
    var data = ctx.getImageData(0, 0, w, h);
    var px = data.data;
    var gray = new Uint8ClampedArray(w * h);
    var i, x, y, idx, r, g, b, lum;

    for (i = 0, idx = 0; i < gray.length; i++, idx += 4) {
      r = px[idx]; g = px[idx + 1]; b = px[idx + 2];
      lum = (r * 0.299 + g * 0.587 + b * 0.114);
      gray[i] = lum;
    }

    for (y = 0; y < h; y++) {
      var dark = 0;
      var row = y * w;
      for (x = 0; x < w; x++) if (gray[row + x] < 168) dark++;
      if (dark > w * 0.52) {
        for (x = 0; x < w; x++) {
          var v = gray[row + x];
          if (v > 95 && v < 205) gray[row + x] = 246;
        }
      }
    }

    var block = 40;
    var out = new ImageData(w, h);
    var op = out.data;
    var by, bx, yy, xx, sum, count, avg, thr, gi, p;
    for (by = 0; by < h; by += block) {
      for (bx = 0; bx < w; bx += block) {
        sum = 0; count = 0;
        for (yy = by; yy < Math.min(h, by + block); yy++) {
          for (xx = bx; xx < Math.min(w, bx + block); xx++) {
            sum += gray[yy * w + xx];
            count++;
          }
        }
        avg = sum / Math.max(1, count);
        thr = avg - 16;
        for (yy = by; yy < Math.min(h, by + block); yy++) {
          for (xx = bx; xx < Math.min(w, bx + block); xx++) {
            gi = yy * w + xx;
            p = gray[gi] < thr ? 18 : 250;
            idx = gi * 4;
            op[idx] = op[idx + 1] = op[idx + 2] = p;
            op[idx + 3] = 255;
          }
        }
      }
    }

    var cropX = Math.round(w * 0.07);
    var cropY = Math.round(h * 0.02);
    var cropW = w - cropX - Math.round(w * 0.02);
    var cropH = h - cropY - Math.round(h * 0.03);
    var tmp = document.createElement("canvas");
    tmp.width = w; tmp.height = h;
    tmp.getContext("2d").putImageData(out, 0, 0);
    var dst = document.createElement("canvas");
    dst.width = cropW; dst.height = cropH;
    dst.getContext("2d").drawImage(tmp, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    return dst;
  }

  function cleanupText(text) {
    if (!text) return "";
    var lines = String(text).split("\n");
    var out = [];
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i]
        .replace(/[|_—–=]{3,}/g, " ")
        .replace(/[ ]{2,}/g, " ")
        .trim();
      if (!line) {
        if (out.length && out[out.length - 1] !== "") out.push("");
        continue;
      }
      if (/^[\W\d]{0,3}$/.test(line)) continue;
      out.push(line);
    }
    return out.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  }

  function ensureWorker(onProgress) {
    if (workerReady) return workerReady;
    workerReady = (async function () {
      var base = new URL("./", document.baseURI).href;
      worker = await Tesseract.createWorker("rus+eng", 1, {
        workerPath: base + "vendor/tesseract/worker.min.js",
        corePath: base + "vendor/tesseract/tesseract-core-simd-lstm.wasm.js",
        langPath: base + "assets/tessdata",
        gzip: true,
        workerBlobURL: false,
        logger: function (m) {
          if (onProgress) onProgress(m);
        }
      });
      await worker.setParameters({
        tessedit_pageseg_mode: "6",
        preserve_interword_spaces: "1"
      });
      return worker;
    })();
    return workerReady;
  }

  async function recognize(canvas, onProgress) {
    var w = await ensureWorker(onProgress);
    var result = await w.recognize(canvas);
    return {
      text: cleanupText(result.data.text || ""),
      raw: result.data.text || "",
      confidence: result.data.confidence || 0,
      words: (result.data.words || []).map(function (wd) {
        return { text: wd.text, conf: wd.confidence, box: wd.bbox };
      })
    };
  }

  function canvasToBlob(canvas, type, quality) {
    return new Promise(function (resolve) {
      canvas.toBlob(function (b) { resolve(b); }, type, quality);
    });
  }

  async function compressOriginal(img, originalBytes) {
    var canvas = drawScaled(img);
    var webp = await canvasToBlob(canvas, "image/webp", 0.42);
    var jpeg = await canvasToBlob(canvas, "image/jpeg", 0.48);
    var pick = webp && webp.size && (!jpeg || webp.size <= jpeg.size) ? webp : jpeg;
    var bytes = pick ? pick.size : 0;
    var ratio = originalBytes ? (1 - bytes / originalBytes) * 100 : 0;
    return {
      blob: pick,
      bytes: bytes,
      originalBytes: originalBytes || 0,
      ratio: Math.max(0, ratio),
      type: pick && pick.type
    };
  }

  function formatBytes(n) {
    if (n < 1024) return n + " Б";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " КБ";
    return (n / (1024 * 1024)).toFixed(2) + " МБ";
  }

  function toMarkdown(meta) {
    var title = meta.title || "Конспект";
    var body = meta.text || "";
    var nl = String.fromCharCode(10);
    return [
      "---",
      "title: " + JSON.stringify(title),
      "engine: Glyph.ai v2.4",
      "source: " + (meta.sourceName || "scan"),
      "confidence: " + (meta.confidence ? meta.confidence.toFixed(1) : "n/a"),
      "compressed: " + (meta.ratio ? meta.ratio.toFixed(1) + "%" : "n/a"),
      "---",
      "",
      "# " + title,
      "",
      body,
      ""
    ].join(nl);
  }

  function downloadBlob(blob, filename) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  global.GlyphEngine = {
    loadImage: loadImage,
    fileToUrl: fileToUrl,
    drawScaled: drawScaled,
    preprocessCanvas: preprocessCanvas,
    cleanupText: cleanupText,
    ensureWorker: ensureWorker,
    recognize: recognize,
    compressOriginal: compressOriginal,
    formatBytes: formatBytes,
    toMarkdown: toMarkdown,
    downloadBlob: downloadBlob
  };
})(window);
