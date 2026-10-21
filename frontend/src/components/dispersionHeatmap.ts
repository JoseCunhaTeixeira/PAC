import { gistSternR } from "./colormaps";

/** A dispersion image's heatmap at its native resolution (a column a frequency, a row a
 * velocity, the fastest at the top), each frequency scaled to its own maximum, squared, in the
 * colours PAC draws dispersion images with. */
export function dispersionHeatmap(fvMap: number[][]): HTMLCanvasElement {
  const nf = fvMap.length;
  const nv = nf > 0 ? fvMap[0].length : 0;
  const off = document.createElement("canvas");
  off.width = nf;
  off.height = nv;
  const octx = off.getContext("2d");
  if (octx && nf > 0 && nv > 0) {
    const imgData = octx.createImageData(nf, nv);
    for (let i = 0; i < nf; i++) {
      const row = fvMap[i];
      let max = 0;
      for (let j = 0; j < nv; j++) max = Math.max(max, row[j]);
      if (max <= 0) max = 1;
      for (let j = 0; j < nv; j++) {
        const [r, g, b] = gistSternR((row[j] / max) ** 2);
        const idx = ((nv - 1 - j) * nf + i) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    octx.putImageData(imgData, 0, 0);
  }
  return off;
}

/** A small dispersion image's heatmap (each frequency already scaled 0 to 255; rows the
 * velocities from the lowest, columns the frequencies), the fastest at the top, squared, in the
 * same colours. */
export function thumbHeatmap(image: number[][]): HTMLCanvasElement {
  const nv = image.length;
  const nf = nv > 0 ? image[0].length : 0;
  const off = document.createElement("canvas");
  off.width = nf;
  off.height = nv;
  const octx = off.getContext("2d");
  if (octx && nf > 0 && nv > 0) {
    const imgData = octx.createImageData(nf, nv);
    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nf; i++) {
        const [r, g, b] = gistSternR((image[j][i] / 255) ** 2);
        const idx = ((nv - 1 - j) * nf + i) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    octx.putImageData(imgData, 0, 0);
  }
  return off;
}
