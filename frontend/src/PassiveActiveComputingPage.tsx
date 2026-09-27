import { ComputingPage } from "./components/computing";
import { LayersIcon } from "./components/icons";
import { ConfigForm } from "./PassiveActiveConfigForm";

export default function PassiveActiveComputingPage() {
  return (
    <ComputingPage
      mode="passive-active"
      title="Passive-active computing"
      subtitle="Compute dispersion images from active shots, as if they were ambient noise"
      icon={<LayersIcon size={24} />}
      art="passive-active"
      needsSources
      Form={ConfigForm}
    />
  );
}
