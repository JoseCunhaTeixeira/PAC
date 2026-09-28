import { ComputingPage } from "./components/computing";
import { PassiveActiveIcon } from "./components/icons";
import { ConfigForm } from "./PassiveActiveConfigForm";

export default function PassiveActiveComputingPage() {
  return (
    <ComputingPage
      mode="passive-active"
      title="Passive-active computing"
      subtitle="Compute dispersion images from active shots, as if they were ambient noise"
      icon={<PassiveActiveIcon size={24} />}
      art="passive-active"
      needsSources
      Form={ConfigForm}
    />
  );
}
