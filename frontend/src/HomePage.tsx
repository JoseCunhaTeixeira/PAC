export default function HomePage() {
  return (
    <div style={{ padding: 24 }}>
      <h1>Home</h1>

      <h2>
        PAC — Passive and Active Computation of Multichannel Analysis of Surface Waves
      </h2>

      <p>👋 Welcome to <strong>PAC</strong>!</p>
      <p>
        📚 This app allows you to perform inline surface wave dispersion analysis using both
        traffic-induced seismic noise and conventional active sources.
      </p>
      <p>
        The surface wave processing workflow is based on the methodology outlined in{" "}
        <a href="https://doi.org/10.26443/seismica.v4i1.1150" target="_blank" rel="noreferrer">
          Cunha Teixeira et al. (2024)
        </a>
        . The inversion of the dispersion curves is performed using the MCMC package{" "}
        <a href="https://bayes-bay.readthedocs.io/en/latest/#" target="_blank" rel="noreferrer">
          BayesBay
        </a>
        , and forward modeling package{" "}
        <a href="https://github.com/keurfonluu/disba" target="_blank" rel="noreferrer">
          Disba
        </a>
        .
      </p>

      <div
        style={{
          background: "var(--accent-soft)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-sm)",
          padding: "12px 16px",
          marginTop: 16,
        }}
      >
        ⓘ For more information, please visit our{" "}
        <a href="https://github.com/JoseCunhaTeixeira/PAC" target="_blank" rel="noreferrer">
          GitHub repository
        </a>
        .
      </div>

      <div
        style={{
          background: "#ffffff00",
          borderRadius: "var(--radius)",
          padding: 24,
          marginTop: 32,
        }}
      >
        <img
          src="/src/assets/logo.png"
          alt="Illustration of train-induced seismic surface waves recorded by sensors along a railway embankment"
          style={{ width: "100%", maxWidth: 560, display: "block", margin: "0 auto" }}
        />
        <img
          src="/src/assets/logo2.png"
          alt="Partner and funding organization logos: Sorbonne Universite, METIS UMR 7619, Mines Paris PSL, SNCF Reseau, European Union, Europe's Rail"
          style={{ width: "100%", maxWidth: 640, display: "block", margin: "32px auto 0" }}
        />
      </div>
    </div>
  );
}
