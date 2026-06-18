import { useEffect, useState } from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import HomePage from "./HomePage";
import ActiveComputingPage from "./ActiveComputingPage";
import PassiveComputingPage from "./PassiveComputingPage";
import PassiveActiveComputingPage from "./PassiveActiveComputingPage";
import DispersionPickingPage from "./DispersionPickingPage";
import { applyTheme, getInitialTheme, ThemeContext, type Theme } from "./theme";
import { CrosshairIcon, HomeIcon, LayersIcon, LogoMark, MoonIcon, SunIcon, WavesIcon, ZapIcon } from "./components/icons";

const NAV_ITEMS = [
  { to: "/", end: true, label: "Home", icon: <HomeIcon /> },
  { to: "/active", end: false, label: "Active Computing", icon: <ZapIcon /> },
  { to: "/passive", end: false, label: "Passive Computing", icon: <WavesIcon /> },
  { to: "/passive-active", end: false, label: "Passive-Active Computing", icon: <LayersIcon /> },
  { to: "/dispersion_picking", end: false, label: "Dispersion Picking", icon: <CrosshairIcon /> },
];

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => getInitialTheme());

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <ThemeContext.Provider value={theme}>
      <div style={{ display: "flex", minHeight: "100vh" }}>
        <aside
          style={{
            width: 270,
            flexShrink: 0,
            position: "sticky",
            top: 0,
            alignSelf: "flex-start",
            height: "100vh",
            display: "flex",
            flexDirection: "column",
            borderRight: "1px solid var(--border)",
            background: "var(--surface)",
            padding: "20px 12px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 10px", marginBottom: 28 }}>
            <LogoMark />
            <span style={{ fontWeight: 700, fontSize: "1.05rem", letterSpacing: "-0.01em", color: "var(--text)" }}>
              PAC
            </span>
          </div>

          <nav style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => "sidebar-link" + (isActive ? " active" : "")}
              >
                {item.icon}
                <span>{item.label}</span>
              </NavLink>
            ))}
          </nav>

          <button
            className="sidebar-toggle"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? <SunIcon /> : <MoonIcon />}
            <span>{theme === "dark" ? "Light mode" : "Dark mode"}</span>
          </button>
        </aside>

        <main style={{ flex: 1, minWidth: 0, width: "100%", maxWidth: 900, margin: "0 auto" }}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/active" element={<ActiveComputingPage />} />
            <Route path="/passive" element={<PassiveComputingPage />} />
            <Route path="/passive-active" element={<PassiveActiveComputingPage />} />
            <Route path="/dispersion_picking" element={<DispersionPickingPage />} />
          </Routes>
        </main>
      </div>
    </ThemeContext.Provider>
  );
}
