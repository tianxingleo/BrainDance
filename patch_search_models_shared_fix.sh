sed -i '87a\
\
    return rows.map((row) => {\
      const modelId = typeof row.id === "string" ? row.id : "";' supabase/functions/search-models/shared.ts
