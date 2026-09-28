import { ConfigForm } from "./ActiveConfigForm";
import { ComputingPage } from "./components/computing";
import { ActiveIcon } from "./components/icons";

export default function ActiveComputingPage() {
  return (
    <ComputingPage
      mode="active"
      title="Active computing"
      subtitle="Compute dispersion images from active shots"
      icon={<ActiveIcon size={24} />}
      art="active"
      needsSources
      Form={ConfigForm}
    />
  );
}
