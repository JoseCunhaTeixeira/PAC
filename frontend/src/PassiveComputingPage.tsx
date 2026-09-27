import { ComputingPage } from "./components/computing";
import { WavesIcon } from "./components/icons";
import { ConfigForm } from "./PassiveConfigForm";

export default function PassiveComputingPage() {
  return (
    <ComputingPage
      mode="passive"
      title="Passive computing"
      subtitle="Compute dispersion images from ambient noise"
      icon={<WavesIcon size={24} />}
      art="passive"
      needsSources={false}
      Form={ConfigForm}
    />
  );
}
