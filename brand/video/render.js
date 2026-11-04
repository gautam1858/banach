// Render index.html frame by frame and encode with ffmpeg.
// Usage: node render.js OUT.mp4 [WIDTH HEIGHT FPS] [--stills t1,t2,...] [--audio AUDIO.wav]
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');

const args = process.argv.slice(2);
const out = args[0];
const W = +(args[1] || 1920), H = +(args[2] || 1080), FPS = +(args[3] || 30);
const stillsIdx = args.indexOf('--stills');
const stills = stillsIdx >= 0 ? args[stillsIdx + 1].split(',').map(Number) : null;
const audioIdx = args.indexOf('--audio');
const audio = audioIdx >= 0 ? args[audioIdx + 1] : null;

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  const url = 'file://' + path.join(__dirname, 'index.html') + `?w=${W}&h=${H}&static=1`;
  await page.goto(url);
  await page.evaluate(() => document.fonts.ready);
  const canvas = await page.$('canvas');
  if (stills) {
    for (const t of stills) {
      await page.evaluate(tt => window.render(tt), t);
      await canvas.screenshot({ path: `${out}-${String(t).replace('.', '_')}.png` });
    }
    await browser.close();
    return;
  }
  const duration = await page.evaluate(() => window.DURATION);
  const frames = Math.round(duration * FPS);
  const ff = ['-y', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-'];
  if (audio) ff.push('-i', audio);
  ff.push('-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart');
  if (audio) ff.push('-c:a', 'aac', '-b:a', '192k', '-shortest');
  ff.push(out);
  const enc = spawn('ffmpeg', ff, { stdio: ['pipe', 'ignore', 'inherit'] });
  for (let i = 0; i < frames; i++) {
    await page.evaluate(tt => window.render(tt), i / FPS);
    const buf = await canvas.screenshot({ type: 'png' });
    if (!enc.stdin.write(buf)) await new Promise(r => enc.stdin.once('drain', r));
    if (i % (FPS * 5) === 0) process.stderr.write(`frame ${i}/${frames}\n`);
  }
  enc.stdin.end();
  await new Promise(r => enc.on('close', r));
  await browser.close();
})();
