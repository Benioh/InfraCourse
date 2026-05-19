"use client";

import { useEffect } from "react";
import { setPageContext, type PageContext } from "@/lib/page-context";

/** Server pages can drop this in to register a snapshot for the AI tutor.
 * On unmount the context is cleared so it doesn't leak across navigations. */
export default function RegisterPageContext({ ctx }: { ctx: PageContext }) {
  // Stable JSON key avoids unnecessary resets when callers pass an object literal.
  const key = JSON.stringify(ctx);
  useEffect(() => {
    setPageContext(ctx);
    return () => {
      setPageContext(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return null;
}
