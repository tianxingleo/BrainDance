export const notifyFlutter = (payload) => {
  if (!window.BrainDanceChannel) return;
  window.BrainDanceChannel.postMessage(JSON.stringify(payload));
};
