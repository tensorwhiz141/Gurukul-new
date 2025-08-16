import { useContext } from "react";
import { TimeContext } from "../context/TimeContextDef";

export const useTimeTracking = () => {
  const context = useContext(TimeContext);
  if (!context) {
    throw new Error("useTimeTracking must be used within a TimeProvider");
  }
  return context;
};
