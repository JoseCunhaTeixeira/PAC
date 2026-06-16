import { Routes, Route, Link } from "react-router-dom";
import ActiveComputingPage from "./active_computing_page/ActiveComputingPage";
import PassiveComputingPage from "./passive_computing_page/PassiveComputingPage";

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
      </nav>

      <Routes>
        <Route path="/" element={<ActiveComputingPage />} />
        <Route path="/active" element={<ActiveComputingPage />} />
        <Route path="/passive" element={<PassiveComputingPage />} />
      </Routes>
    </>
  );
}