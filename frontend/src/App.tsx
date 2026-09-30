import { useEffect, useState, type ReactNode } from "react";
import { Routes, Route, NavLink, Link, useLocation } from "react-router-dom";
import HomePage from "./HomePage";
import ActiveComputingPage from "./ActiveComputingPage";
import PassiveComputingPage from "./PassiveComputingPage";
import PassiveActiveComputingPage from "./PassiveActiveComputingPage";
import DispersionPickingPage from "./DispersionPickingPage";
import InversionPage from "./InversionPage";
import PetroInversionPage from "./PetroInversionPage";
import VisualizationPage from "./VisualizationPage";
import ChatPage from "./ChatPage";
import { API } from "./api";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { useStayPut } from "./components/stayPut";
import { TipLayer } from "./components/TipLayer";
import { runsAt, useRunning } from "./components/running";
import { applyTheme, getInitialTheme, ThemeContext, type Theme } from "./theme";
import {
  ActiveIcon,
  CrosshairIcon,
  DepthIcon,
  EyeIcon,
  HomeIcon,
  MoonIcon,
  OutcropIcon,
  PassiveActiveIcon,
  PassiveIcon,
  SparklesIcon,
  SunIcon,
} from "./components/icons";
// PAC's version in the sidebar: the frontend's own, kept in step with pyproject.toml.
import packageJson from "../package.json";
import logoDeepWaveLight from "./assets/logo_DeepWave_lightmode.png";
import logoDeepWaveDark from "./assets/logo_DeepWave_darkmode.png";

interface NavItem {
  to: string;
  label: string;
  icon: ReactNode;
}

// The sidebar: home, the AI assistant who can do it all (where PAC was installed with PACo),
// then the workflow's order: records to images, images to curves and models, what came out.
const NAV_GROUPS: { label: string | null; items: NavItem[] }[] = [
  { label: null, items: [{ to: "/", label: "Home", icon: <HomeIcon /> }] },
  { label: "Automate", items: [{ to: "/assistant", label: "AI assistant", icon: <SparklesIcon /> }] },
  {
    label: "Compute",
    items: [
      { to: "/active", label: "Active", icon: <ActiveIcon /> },
      { to: "/passive", label: "Passive", icon: <PassiveIcon /> },
      { to: "/passive-active", label: "Passive-active", icon: <PassiveActiveIcon /> },
    ],
  },
  {
    label: "Analyse",
    items: [
      { to: "/dispersion_picking", label: "Dispersion picking", icon: <CrosshairIcon /> },
      { to: "/seismic_inversion", label: "Seismic inversion", icon: <DepthIcon /> },
      { to: "/petro_inversion", label: "Petrophysical inversion", icon: <OutcropIcon /> },
    ],
  },
  {
    label: "Review",
    items: [{ to: "/visualization", label: "Visualization", icon: <EyeIcon /> }],
  },
];

export default function App() {
  const location = useLocation();
  const [theme, setTheme] = useState<Theme>(() => getInitialTheme());
  // The assistant's page, only where PAC was installed with it.
  const [assistant, setAssistant] = useState(false);
  // What runs, beside each page in the menu.
  const running = useRunning(assistant);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);
  // What a click presses stays where it is on screen while the page changes around it.
  useStayPut();

  useEffect(() => {
    fetch(`${API}/agent/installed`)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data: { installed: boolean }) => setAssistant(data.installed))
      .catch(() => setAssistant(false));
  }, []);

  return (
    <ThemeContext.Provider value={theme}>
      <div className="app">
        <aside className="sidebar">
          <Link to="/" className="sidebar-brand">
            <img src={theme === "dark" ? logoDeepWaveDark : logoDeepWaveLight} alt="DeepWave logo" />
            <div>
              <strong>PAC</strong>
              <small>Surface-wave imaging</small>
            </div>
          </Link>

          <nav className="sidebar-nav">
            {NAV_GROUPS.map((group) => {
              const items = group.items.filter((item) => assistant || item.to !== "/assistant");
              if (items.length === 0) return null; // the assistant's group, without PACo
              return (
                <div key={group.label ?? "top"} className="sidebar-group">
                  {group.label && <div className="sidebar-group-label">{group.label}</div>}
                  {items.map((item) => (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      end={item.to === "/"}
                      title={item.label}
                      className={({ isActive }) => "sidebar-link" + (isActive ? " active" : "")}
                    >
                      {item.icon}
                      <span>{item.label}</span>
                      {runsAt(item.to, running) && (
                        <i
                          className="sidebar-running"
                          aria-label="Running"
                          data-tip={item.to === "/assistant" ? "Answering" : "Running"}
                        />
                      )}
                    </NavLink>
                  ))}
                </div>
              );
            })}
          </nav>

          <div className="sidebar-footer">
            <span>v{packageJson.version}</span>
            <div className="theme-switch" role="group" aria-label="Theme">
              <button
                type="button"
                title="Light mode"
                aria-pressed={theme === "light"}
                className={theme === "light" ? "active" : ""}
                onClick={() => setTheme("light")}
              >
                <SunIcon size={15} />
              </button>
              <button
                type="button"
                title="Dark mode"
                aria-pressed={theme === "dark"}
                className={theme === "dark" ? "active" : ""}
                onClick={() => setTheme("dark")}
              >
                <MoonIcon size={15} />
              </button>
            </div>
          </div>
        </aside>

        <TipLayer />
        <main className="app-main">
          <ErrorBoundary resetKey={location.pathname}>
          <Routes>
            <Route path="/" element={<HomePage assistant={assistant} />} />
            <Route path="/active" element={<ActiveComputingPage />} />
            <Route path="/passive" element={<PassiveComputingPage />} />
            <Route path="/passive-active" element={<PassiveActiveComputingPage />} />
            <Route path="/dispersion_picking" element={<DispersionPickingPage />} />
            <Route path="/seismic_inversion" element={<InversionPage />} />
            <Route path="/petro_inversion" element={<PetroInversionPage />} />
            <Route path="/visualization" element={<VisualizationPage />} />
            <Route path="/assistant" element={<ChatPage />} />
          </Routes>
          </ErrorBoundary>
        </main>
      </div>
    </ThemeContext.Provider>
  );
}
