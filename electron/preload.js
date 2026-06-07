const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('riff', {
  start: () => ipcRenderer.invoke('agent:start'),
  stop: () => ipcRenderer.invoke('agent:stop'),
  running: () => ipcRenderer.invoke('agent:running'),
  port: () => ipcRenderer.invoke('agent:port'),
  onLog: (cb) => ipcRenderer.on('py:log', (_e, payload) => cb(payload)),
});
