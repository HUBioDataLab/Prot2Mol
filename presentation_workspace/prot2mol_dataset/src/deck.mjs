import fs from "node:fs/promises";
import path from "node:path";
import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  layers,
  panel,
  text,
  shape,
  chart,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  fr,
} from "@oai/artifact-tool";

const W = 1920;
const H = 1080;
const OUT = "output/output.pptx";
const PREVIEW_DIR = "scratch/previews";

const C = {
  bg: "#F7F8F4",
  ink: "#102027",
  slate: "#253B40",
  muted: "#627276",
  faint: "#DCE3DD",
  faint2: "#EDF0EA",
  white: "#FFFFFF",
  teal: "#0F766E",
  blue: "#2563EB",
  green: "#15803D",
  amber: "#B7791F",
  red: "#B42318",
};

const fmt = (n) => new Intl.NumberFormat("en-US").format(n);
const m = (n, d = 2) => `${(n / 1_000_000).toFixed(d)}M`;
const pct = (part, total) => `${((part / total) * 100).toFixed(1)}%`;

const data = {
  boltz: {
    binaryRows: 5_581_718,
    binaryPos: 1_667_556,
    binaryNeg: 3_914_162,
    continuousRows: 3_720_045,
    proteins: 13_806,
    proteinSequences: 13_843,
    smiles: 2_009_556,
    clusters: 10_718,
    splitRows: { train: 3_906_708, val: 820_033, test: 854_977 },
    clusterCounts: { train: 10_650, val: 21, test: 29 },
    rankingPairs: { train: 4_000_000, val: 500_000, test: 500_000 },
    longProteinRows: 890_525,
    sources: [
      ["BindingDB", 1_817_178, "curated binding measurements; thresholded binary + ranking"],
      ["PubChem HTS", 1_375_954, "screen labels, mostly negative"],
      ["CeMM", 1_336_960, "proteome-wide compound screen labels"],
      ["ChEMBL", 1_042_315, "curated single-protein bioactivity"],
      ["MIDAS", 9_311, "target engagement labels"],
    ],
  },
  papyrus: {
    rows: 1_053_052,
    pos: 419_341,
    neg: 633_711,
    targets: 4_271,
    proteinSequences: 4_259,
    smiles: 581_696,
    clusters: 3_466,
    splitRows: { train: 776_137, val: 139_583, test: 137_332 },
    clusterCounts: { train: 3_446, val: 10, test: 10 },
    rankingPairs: { train: 4_000_000, val: 500_000, test: 500_000 },
    lengthMax: 999,
    rawSources: [
      ["ChEMBL30", 835_250, "main curated bioactivity contribution"],
      ["ExCAPE-DB", 212_506, "large public bioactivity benchmark source"],
      ["Merget2017", 72_526, "chemogenomic target-compound data"],
      ["Christmann2016", 69_407, "chemogenomic target-compound data"],
      ["Sharma2016", 46_035, "kinase-focused activity matrix"],
      ["Klaeger2017", 3_111, "kinase profiling contribution"],
    ],
  },
};

function t(value, opts = {}) {
  return text(value, {
    name: opts.name,
    width: opts.width ?? fill,
    height: opts.height ?? hug,
    columnSpan: opts.columnSpan,
    rowSpan: opts.rowSpan,
    style: {
      fontFamily: "Aptos",
      fontSize: opts.size ?? 24,
      bold: opts.bold ?? false,
      color: opts.color ?? C.ink,
      horizontalAlignment: opts.align,
    },
  });
}

function slide(presentation, children) {
  const s = presentation.slides.add();
  s.compose(
    layers({ width: fill, height: fill }, [
      shape({ width: fill, height: fill, fill: C.bg }),
      column({ width: fill, height: fill, padding: { x: 86, y: 66 }, gap: 28 }, children),
    ]),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
  return s;
}

function title(kicker, headline, subhead) {
  return column({ width: fill, height: hug, gap: 10 }, [
    t(kicker.toUpperCase(), { size: 17, bold: true, color: C.teal }),
    t(headline, { name: "slide-title", size: 54, bold: true, color: C.ink }),
    subhead ? t(subhead, { name: "slide-subtitle", size: 24, color: C.muted, width: wrap(1380) }) : null,
  ].filter(Boolean));
}

function metric(label, value, note, color = C.teal) {
  return column({ width: fill, height: hug, gap: 5 }, [
    t(value, { size: 46, bold: true, color }),
    t(label, { size: 20, bold: true, color: C.ink }),
    note ? t(note, { size: 16, color: C.muted }) : null,
  ].filter(Boolean));
}

function tableBlock(name, headers, rows, widths, opts = {}) {
  const cols = widths.map((w) => (typeof w === "number" ? fixed(w) : fr(w)));
  const header = grid(
    { width: fill, height: hug, columns: cols, columnGap: 16 },
    headers.map((h) => t(h, { size: opts.headerSize ?? 15, bold: true, color: C.teal })),
  );
  const body = rows.flatMap((r, idx) => {
    const rowNode = grid(
      { width: fill, height: hug, columns: cols, columnGap: 16, padding: { x: 0, y: 2 } },
      r.map((cell, cidx) =>
        t(String(cell), {
          size: opts.bodySize ?? 18,
          bold: cidx === 0,
          color: cidx === 0 ? C.ink : C.slate,
        }),
      ),
    );
    return idx === rows.length - 1
      ? [rowNode]
      : [rowNode, rule({ width: fill, stroke: C.faint2, weight: 1 })];
  });
  return panel(
    {
      name,
      width: opts.width ?? fill,
      height: opts.height ?? hug,
      fill: C.white,
      line: { style: "solid", width: 1, fill: C.faint },
      borderRadius: "rounded-md",
      padding: opts.padding ?? { x: 24, y: 22 },
      columnSpan: opts.columnSpan,
      rowSpan: opts.rowSpan,
    },
    column({ width: fill, height: hug, gap: 12 }, [header, rule({ width: fill, stroke: C.faint, weight: 1 }), ...body]),
  );
}

function tag(textValue, color, bg) {
  return panel(
    {
      width: hug,
      height: hug,
      fill: bg,
      line: { style: "solid", width: 0, fill: bg },
      borderRadius: "rounded-full",
      padding: { x: 14, y: 7 },
    },
    t(textValue, { width: hug, size: 16, bold: true, color }),
  );
}

async function saveBlob(blob, filePath) {
  const ab = await blob.arrayBuffer();
  await fs.writeFile(filePath, Buffer.from(ab));
}

function buildDeck() {
  const p = Presentation.create({ slideSize: { width: W, height: H } });

  slide(p, [
    title(
      "Where the data comes from",
      "Two dataset families: Boltz-style reconstruction and Papyrus",
      "Boltz-style keeps assay/source provenance for each row; Papyrus gives a compact high-quality aggregated pChEMBL median table.",
    ),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1)], columnGap: 40 }, [
      tableBlock(
        "boltz-source-table",
        ["Boltz-style source", "Rows", "Contribution"],
        data.boltz.sources.map(([s, n, c]) => [s, fmt(n), c]),
        [180, 135, 520],
        { bodySize: 17, rowSpan: 2 },
      ),
      tableBlock(
        "papyrus-source-table",
        ["Papyrus raw source", "Rows", "Contribution"],
        data.papyrus.rawSources.map(([s, n, c]) => [s, fmt(n), c]),
        [190, 130, 500],
        { bodySize: 17, rowSpan: 2 },
      ),
    ]),
    row({ width: fill, height: hug, justify: "between", align: "center" }, [
      row({ width: hug, height: hug, gap: 10 }, [
        tag("Boltz-style: ChEMBL + BindingDB + screens", C.blue, "#EAF1FF"),
        tag("Papyrus: all local rows Quality=High", C.green, "#E8F3E8"),
      ]),
      t("Source counts are from the prepared local files.", { width: hug, size: 16, color: C.muted }),
    ]),
  ]);

  slide(p, [
    title(
      "What each dataset contains",
      "The datasets answer different training questions",
      "Use the mixed Boltz-style dataset for binary hit prediction and assay-aware evaluation; use Papyrus as auxiliary high-quality pChEMBL median data.",
    ),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1)], columnGap: 42 }, [
      panel(
        { width: fill, height: fill, fill: C.white, line: { style: "solid", width: 1, fill: C.faint }, padding: 28 },
        column({ width: fill, height: fill, gap: 22 }, [
          t("Boltz-style dataset", { size: 31, bold: true, color: C.blue }),
          row({ width: fill, height: hug, gap: 24 }, [
            metric("binary rows", m(data.boltz.binaryRows), `${pct(data.boltz.binaryPos, data.boltz.binaryRows)} positive`, C.blue),
            metric("continuous affinity rows", m(data.boltz.continuousRows), "ChEMBL + BindingDB", C.teal),
          ]),
          tableBlock(
            "boltz-contains",
            ["View", "Contains"],
            [
              ["Binary all-source", "BindingDB, ChEMBL, PubChem HTS, CeMM, MIDAS"],
              ["Screen-only binary", "PubChem HTS, CeMM, MIDAS"],
              ["Threshold binary", "ChEMBL/BindingDB pChEMBL >= 6.0"],
              ["Ranking", "same-protein ligand pairs from continuous affinity"],
              ["Caveat", `${fmt(data.boltz.longProteinRows)} all-source rows have protein length >1000`],
            ],
            [210, 610],
            { bodySize: 18, padding: { x: 18, y: 16 } },
          ),
        ]),
      ),
      panel(
        { width: fill, height: fill, fill: C.white, line: { style: "solid", width: 1, fill: C.faint }, padding: 28 },
        column({ width: fill, height: fill, gap: 22 }, [
          t("Papyrus dataset", { size: 31, bold: true, color: C.green }),
          row({ width: fill, height: hug, gap: 24 }, [
            metric("rows", m(data.papyrus.rows), `${pct(data.papyrus.pos, data.papyrus.rows)} positive at pChEMBL >= 6`, C.green),
            metric("protein length max", String(data.papyrus.lengthMax), "prepared with length <1000", C.amber),
          ]),
          tableBlock(
            "papyrus-contains",
            ["View", "Contains"],
            [
              ["Affinity", "aggregated pChEMBL median per target-compound"],
              ["Binary", "pChEMBL median >= 6.0 threshold"],
              ["Ranking", "same-protein ligand pairs from Papyrus medians"],
              ["Provenance", "original raw sources kept in summary, prepared rows tagged Papyrus"],
              ["Caveat", "less assay-level detail than ChEMBL/BindingDB export"],
            ],
            [210, 610],
            { bodySize: 18, padding: { x: 18, y: 16 } },
          ),
        ]),
      ),
    ]),
  ]);

  slide(p, [
    title(
      "Statistics",
      "Boltz-style is broader; Papyrus is cleaner and smaller",
      "The main tradeoff is coverage versus aggregation: Boltz-style keeps screen negatives and assay metadata, while Papyrus keeps high-quality median pChEMBL rows.",
    ),
    grid({ width: fill, height: fill, columns: [fr(1.05), fr(0.95)], columnGap: 44 }, [
      tableBlock(
        "stats-table",
        ["Metric", "Boltz-style", "Papyrus"],
        [
          ["binary rows", fmt(data.boltz.binaryRows), fmt(data.papyrus.rows)],
          ["positive rows", `${fmt(data.boltz.binaryPos)} (${pct(data.boltz.binaryPos, data.boltz.binaryRows)})`, `${fmt(data.papyrus.pos)} (${pct(data.papyrus.pos, data.papyrus.rows)})`],
          ["negative rows", fmt(data.boltz.binaryNeg), fmt(data.papyrus.neg)],
          ["continuous affinity rows", fmt(data.boltz.continuousRows), fmt(data.papyrus.rows)],
          ["protein sequences", fmt(data.boltz.proteinSequences), fmt(data.papyrus.proteinSequences)],
          ["unique SMILES", fmt(data.boltz.smiles), fmt(data.papyrus.smiles)],
          ["ranking pairs", "4M / 500k / 500k", "4M / 500k / 500k"],
        ],
        [280, 250, 250],
        { bodySize: 20, headerSize: 16 },
      ),
      panel(
        { width: fill, height: fill, fill: C.white, line: { style: "solid", width: 1, fill: C.faint }, padding: { x: 18, y: 16 } },
        chart({
          name: "dataset-size-chart",
          chartType: "bar",
          width: fill,
          height: fill,
          config: {
            title: "Dataset size comparison, millions",
            categories: ["Binary rows", "Protein seqs", "Unique SMILES"],
            series: [
              {
                name: "Boltz-style",
                values: [
                  Number((data.boltz.binaryRows / 1_000_000).toFixed(3)),
                  Number((data.boltz.proteinSequences / 1_000_000).toFixed(3)),
                  Number((data.boltz.smiles / 1_000_000).toFixed(3)),
                ],
              },
              {
                name: "Papyrus",
                values: [
                  Number((data.papyrus.rows / 1_000_000).toFixed(3)),
                  Number((data.papyrus.proteinSequences / 1_000_000).toFixed(3)),
                  Number((data.papyrus.smiles / 1_000_000).toFixed(3)),
                ],
              },
            ],
          },
        }),
      ),
    ]),
  ]);

  slide(p, [
    title(
      "Clustering and split",
      "Both datasets use 90% MMseqs protein-cluster splits",
      "Ranking pairs are generated after split assignment, deduped by protein+winner+loser key, and verified for zero train/val/test overlap.",
    ),
    grid({ width: fill, height: fill, columns: [fr(1), fr(1)], columnGap: 42 }, [
      panel(
        { width: fill, height: fill, fill: C.white, line: { style: "solid", width: 1, fill: C.faint }, padding: 26 },
        column({ width: fill, height: fill, gap: 20 }, [
          t("Boltz-style split", { size: 30, bold: true, color: C.blue }),
          row({ width: fill, height: hug, gap: 22 }, [
            metric("clusters", fmt(data.boltz.clusters), "90% sequence identity", C.blue),
            metric("leakage", "0", "cluster and pair-key overlap", C.green),
          ]),
          tableBlock(
            "boltz-split-table",
            ["Split", "Rows", "Clusters", "Ranking pairs"],
            [
              ["Train", fmt(data.boltz.splitRows.train), fmt(data.boltz.clusterCounts.train), fmt(data.boltz.rankingPairs.train)],
              ["Val", fmt(data.boltz.splitRows.val), fmt(data.boltz.clusterCounts.val), fmt(data.boltz.rankingPairs.val)],
              ["Test", fmt(data.boltz.splitRows.test), fmt(data.boltz.clusterCounts.test), fmt(data.boltz.rankingPairs.test)],
            ],
            [145, 190, 150, 210],
            { bodySize: 20, padding: { x: 18, y: 16 } },
          ),
        ]),
      ),
      panel(
        { width: fill, height: fill, fill: C.white, line: { style: "solid", width: 1, fill: C.faint }, padding: 26 },
        column({ width: fill, height: fill, gap: 20 }, [
          t("Papyrus split", { size: 30, bold: true, color: C.green }),
          row({ width: fill, height: hug, gap: 22 }, [
            metric("clusters", fmt(data.papyrus.clusters), "90% sequence identity", C.green),
            metric("leakage", "0", "cluster and pair-key overlap", C.green),
          ]),
          tableBlock(
            "papyrus-split-table",
            ["Split", "Rows", "Clusters", "Ranking pairs"],
            [
              ["Train", fmt(data.papyrus.splitRows.train), fmt(data.papyrus.clusterCounts.train), fmt(data.papyrus.rankingPairs.train)],
              ["Val", fmt(data.papyrus.splitRows.val), fmt(data.papyrus.clusterCounts.val), fmt(data.papyrus.rankingPairs.val)],
              ["Test", fmt(data.papyrus.splitRows.test), fmt(data.papyrus.clusterCounts.test), fmt(data.papyrus.rankingPairs.test)],
            ],
            [145, 190, 150, 210],
            { bodySize: 20, padding: { x: 18, y: 16 } },
          ),
        ]),
      ),
    ]),
  ]);

  return p;
}

async function main() {
  await fs.mkdir("output", { recursive: true });
  await fs.rm(PREVIEW_DIR, { recursive: true, force: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });

  const presentation = buildDeck();
  const pptxBlob = await PresentationFile.exportPptx(presentation);
  await pptxBlob.save(OUT);

  for (let i = 0; i < presentation.slides.count; i += 1) {
    const slide = presentation.slides.getItem(i);
    const png = await slide.export({ format: "png", scale: 1 });
    await saveBlob(png, path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`));
  }
  console.log(`Exported ${OUT}`);
  console.log(`Rendered ${presentation.slides.count} previews to ${PREVIEW_DIR}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
