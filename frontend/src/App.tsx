import { Routes, Route, Link } from "react-router-dom";
import ActiveComputingPage from "./ActiveComputingPage";
import PassiveComputingPage from "./PassiveComputingPage";
import ActivePassiveComputingPage from "./ActivePassiveComputingPage";

export default function App() {
  return (
    <>
      <nav
        style={{
          padding: 16,
          borderBottom: "1px solid #ccc",
          marginBottom: 24,
        }}
      >
        <Link to="/active">Active Computing</Link>
        {" | "}
        <Link to="/passive">Passive Computing</Link>
        {" | "}
        <Link to="/passive-active">Passive-Active Computing</Link>
      </nav>

      <Routes>
        <Route path="/" element={<ActiveComputingPage />} />
        <Route path="/active" element={<ActiveComputingPage />} />
        <Route path="/passive" element={<PassiveComputingPage />} />
        <Route path="/passive-active" element={<ActivePassiveComputingPage />} />
      </Routes>
    </>
  );
}