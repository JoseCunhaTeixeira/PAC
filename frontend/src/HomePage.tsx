import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { HeroArt } from "./components/HeroArt";
import {
  ArrowRightIcon,
  BookIcon,
  CrosshairIcon,
  DepthIcon,
  EyeIcon,
  FlaskIcon,
  GithubIcon,
  LayersIcon,
  SparklesIcon,
  WavesIcon,
  ZapIcon,
} from "./components/icons";
import fieldIllustration from "./assets/logo.png";
import partners from "./assets/logo2.png";
import logoDeepWaveLight from "./assets/logo_DeepWave_lightmode.png";
import logoDeepWaveDark from "./assets/logo_DeepWave_darkmode.png";
import { useTheme } from "./theme";
import "./home.css";

interface Way {
  to: string;
  icon: ReactNode;
  title: string;
  text: string;
}

// The workflow; its first and third steps come in several kinds, listed under it.
const STEPS: Way[] = [
  { to: "/active", icon: <ZapIcon />, title: "Compute", text: "Records to dispersion images" },
  { to: "/dispersion_picking", icon: <CrosshairIcon />, title: "Pick", text: "Dispersion images to curves" },
  { to: "/seismic_inversion", icon: <DepthIcon />, title: "Invert", text: "Curves to ground models" },
  { to: "/visualization", icon: <EyeIcon />, title: "Review", text: "Every window, check and model" },
];

const COMPUTE: Way[] = [
  { to: "/active", icon: <ZapIcon size={20} />, title: "Active", text: "Hammer or weight-drop shots" },
  { to: "/passive", icon: <WavesIcon size={20} />, title: "Passive", text: "Traffic or ambient noise" },
  { to: "/passive-active", icon: <LayersIcon size={20} />, title: "Passive-active", text: "Shots, correlated like noise" },
];

const INVERT: Way[] = [
  { to: "/seismic_inversion", icon: <DepthIcon size={20} />, title: "Seismic", text: "Vs profiles by MCMC" },
  { to: "/petro_inversion", icon: <FlaskIcon size={20} />, title: "Petrophysical", text: "Soil types and penetration resistance" },
];

const TOOLS = [
  { name: "PACo", role: "The AI assistant and its quality checks", href: "https://github.com/JoseCunhaTeixeira/PACo" },
  { name: "sigpipe", role: "Processing, dispersion, inversion, Silex", href: "https://github.com/JoseCunhaTeixeira/sigpipe" },
  { name: "Disba", role: "Forward modelling", href: "https://github.com/keurfonluu/disba" },
];

function Ways({ title, ways }: { title: string; ways: Way[] }) {
  return (
    <section className="home-section">
      <h2 className="home-title">{title}</h2>
      <div className="modes" style={{ gridTemplateColumns: `repeat(${ways.length}, minmax(0, 1fr))` }}>
        {ways.map((way) => (
          <Link key={way.title} to={way.to} className="mode">
            <span className="mode-icon">{way.icon}</span>
            <div>
              <strong>{way.title}</strong>
              <span className="mode-source">{way.text}</span>
            </div>
            <ArrowRightIcon size={16} />
          </Link>
        ))}
      </div>
    </section>
  );
}

export default function HomePage({ assistant = false }: { assistant?: boolean }) {
  const theme = useTheme();
  return (
    <div className="page home">
      <section className="hero">
        <div className="hero-text">
          <span className="hero-eyebrow">Multichannel Analysis of Surface Waves</span>
          <h1>
            From seismic records
            <br />
            to shear-wave velocity.
          </h1>
          <p>Using active shots or passive traffic noise.</p>
          <div className="hero-actions">
            <Link to="/active" className="button hero-primary">
              Start computing <ArrowRightIcon size={16} />
            </Link>
            <Link to="/visualization" className="button hero-secondary">
              <EyeIcon size={16} /> Explore results
            </Link>
            {assistant && (
              <Link to="/assistant" className="button hero-secondary">
                <SparklesIcon size={16} /> Ask the AI assistant
              </Link>
            )}
          </div>
        </div>
        <div className="hero-visual">
          <HeroArt />
        </div>
      </section>

      <section className="home-section">
        <h2 className="home-title">The workflow</h2>
        <ol className="steps">
          {STEPS.map((step, i) => (
            <li key={step.title}>
              <Link to={step.to} className="step">
                <div className="step-top">
                  <span className="step-icon">{step.icon}</span>
                  <span className="step-number">{String(i + 1).padStart(2, "0")}</span>
                </div>
                <strong>{step.title}</strong>
                <p>{step.text}</p>
              </Link>
            </li>
          ))}
        </ol>
      </section>

      <Ways title="Three ways to compute" ways={COMPUTE} />
      <Ways title="Two ways to invert" ways={INVERT} />

      <section className="home-section">
        <h2 className="home-title">Learn more</h2>
        <div className="about">
          <figure className="field">
            <img src={fieldIllustration} alt="Train-induced surface waves recorded by sensors along a railway embankment" />
          </figure>
          <div className="about-text">
            <ul className="tools">
              {TOOLS.map((tool) => (
                <li key={tool.name}>
                  <a href={tool.href} target="_blank" rel="noreferrer">
                    <strong>{tool.name}</strong>
                    <span>{tool.role}</span>
                  </a>
                </li>
              ))}
            </ul>
            <div className="about-links">
              <a className="button secondary" href="https://github.com/JoseCunhaTeixeira/PAC" target="_blank" rel="noreferrer">
                <GithubIcon size={16} /> PAC on GitHub
              </a>
              <a className="button secondary" href="https://doi.org/10.26443/seismica.v4i1.1150" target="_blank" rel="noreferrer">
                <BookIcon size={16} /> Cunha Teixeira et al. (2024)
              </a>
            </div>
          </div>
        </div>
      </section>

      <footer className="partners">
        {/* DeepWave first, in the theme's version as the sidebar has it, then the partners. */}
        <span className="partners-deepwave">
          <img src={theme === "dark" ? logoDeepWaveDark : logoDeepWaveLight} alt="" />
          <span>DeepWave</span>
        </span>
        <img
          className="partners-others"
          src={partners}
          alt="Partner and funding organization logos: Sorbonne Universite, METIS UMR 7619, Mines Paris PSL, SNCF Reseau, European Union, Europe's Rail"
        />
      </footer>
    </div>
  );
}
