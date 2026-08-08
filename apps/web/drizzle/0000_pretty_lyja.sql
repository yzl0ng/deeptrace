CREATE TABLE `live_rate_limits` (
	`bucket` text PRIMARY KEY NOT NULL,
	`count` integer DEFAULT 0 NOT NULL,
	`updated_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
CREATE TABLE `live_runs` (
	`id` text PRIMARY KEY NOT NULL,
	`query` text NOT NULL,
	`status` text NOT NULL,
	`model` text NOT NULL,
	`trace_json` text NOT NULL,
	`evidence_json` text NOT NULL,
	`final_report` text NOT NULL,
	`usage_json` text NOT NULL,
	`created_at` text DEFAULT CURRENT_TIMESTAMP NOT NULL
);
