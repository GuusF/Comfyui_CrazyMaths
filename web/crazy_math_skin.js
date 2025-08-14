import { app } from "../../scripts/app.js";

const SKIN_CONFIG = {
    nodeTypes: [
        "MathAlphaMask",
        "FractalNoiseAlphaMask",
        "VoronoiAlphaMask",
        "SierpinskiAlphaMask",
        "HarmonographAlphaMask",
        "AttractorAlphaMask",
        "QuasicrystalAlphaMask",
        "EquationAlphaMask"
    ],
    backgroundVideo: "/extensions/Comfyui_CrazyMaths/background_video.mp4",
    opacity: 0.9
};

let backgroundVideo;
function preloadBackgroundVideo() {
    if (backgroundVideo) return;
    backgroundVideo = document.createElement("video");
    backgroundVideo.src = SKIN_CONFIG.backgroundVideo;
    backgroundVideo.autoplay = true;
    backgroundVideo.muted = true;
    backgroundVideo.loop = true;
    backgroundVideo.playsInline = true;
}

function drawVideo(ctx, node) {
    if (!backgroundVideo || backgroundVideo.readyState < 3) return;
    const nw = node.size[0];
    const nh = node.size[1];
    const vw = backgroundVideo.videoWidth;
    const vh = backgroundVideo.videoHeight;
    if (!vw || !vh) return;
    let drawW, drawH;
    const nodeRatio = nw / nh;
    const videoRatio = vw / vh;
    if (videoRatio > nodeRatio) {
        drawH = nh;
        drawW = videoRatio * drawH;
    } else {
        drawW = nw;
        drawH = drawW / videoRatio;
    }
    const dx = (nw - drawW) / 2;
    const dy = (nh - drawH) / 2;

    ctx.save();
    const bleed = 1;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(-bleed, -bleed, nw + 2 * bleed, nh + 2 * bleed, 8);
    else ctx.rect(-bleed, -bleed, nw + 2 * bleed, nh + 2 * bleed);
    ctx.clip();

    ctx.globalAlpha = SKIN_CONFIG.opacity;
    ctx.drawImage(backgroundVideo, dx, dy, drawW, drawH);
    ctx.globalAlpha = 1;
    ctx.restore();
}

app.registerExtension({
    name: "CrazyMaths.Skin",
    async setup() {
        preloadBackgroundVideo();
    },
    async nodeCreated(node) {
        const type = node.comfyClass || node.type;
        if (!SKIN_CONFIG.nodeTypes.includes(type)) return;
        if (node.html) node.html.classList.add("crazy-math-node");
        if (!node._originalOnDrawBackground)
            node._originalOnDrawBackground = node.onDrawBackground;
        node.onDrawBackground = function(ctx) {
            drawVideo(ctx, this);
            if (this._originalOnDrawBackground) {
                this._originalOnDrawBackground.call(this, ctx);
            }
        };
        const loop = () => {
            if (node.graph) {
                node.setDirtyCanvas(true, true);
                requestAnimationFrame(loop);
            }
        };
        requestAnimationFrame(loop);
    }
});

export { SKIN_CONFIG };
