import { ComputingPage } from "./components/computing";
import { PassiveIcon } from "./components/icons";
import { ConfigForm } from "./PassiveConfigForm";

export default function PassiveComputingPage() {
  return (
    <ComputingPage
      mode="passive"
      title="Passive computing"
      subtitle="Compute dispersion images from ambient noise"
      icon={<PassiveIcon size={24} />}
      art="passive"
      needsSources={false}
      Form={ConfigForm}
    />
  );
}
