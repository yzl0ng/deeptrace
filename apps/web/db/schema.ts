import { sql } from "drizzle-orm";
import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const liveRuns = sqliteTable("live_runs", {
  id: text("id").primaryKey(),
  query: text("query").notNull(),
  status: text("status").notNull(),
  model: text("model").notNull(),
  traceJson: text("trace_json").notNull(),
  evidenceJson: text("evidence_json").notNull(),
  finalReport: text("final_report").notNull(),
  usageJson: text("usage_json").notNull(),
  createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});

export const liveRateLimits = sqliteTable("live_rate_limits", {
  bucket: text("bucket").primaryKey(),
  count: integer("count").notNull().default(0),
  updatedAt: text("updated_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});
