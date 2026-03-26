sed -i 's/"open_model", "fly_to_pose", "highlight_hotspot"/"open_model", "fly_to_pose"/' supabase/functions/_shared/agent-core/spatialAgent.ts
sed -i '/type: "highlight_hotspot"/,+9d' supabase/functions/_shared/agent-core/spatialAgent.ts
