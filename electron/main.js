const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const API_PORT = process.env.RIFF_API_PORT || '8765';

let pyProcess = null;
let mainWindow = null;

function pythonBin() {
  const venv = path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
  return require('fs').existsSync(venv) ? venv : 'python3';
}

function startPython() {
  if (pyProcess) return;
  const args = ['voice_agent.py', 'console'];
  pyProcess = spawn(pythonBin(), args, {
    cwd: PROJECT_ROOT,
    env: { ...process.env, RIFF_API_PORT: API_PORT, PYTHONUNBUFFERED: '1' },
  });
  pyProcess.stdout.on('data', (d) => sendLog('stdout', d.toString()));
  pyProcess.stderr.on('data', (d) => sendLog('stderr', d.toString()));
  pyProcess.on('exit', (code) => {
    sendLog('exit', `Python exited with code ${code}`);
    pyProcess = null;
  });
}

function stopPython() {
  if (!pyProcess) return;
  pyProcess.kill('SIGINT');
  pyProcess = null;
}

function sendLog(channel, text) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('py:log', { channel, text });
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 920,
    height: 640,
    title: 'Riff',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  mainWindow.webContents.send('py:port', API_PORT);
}

ipcMain.handle('agent:start', () => { startPython(); return { running: !!pyProcess }; });
ipcMain.handle('agent:stop', () => { stopPython(); return { running: !!pyProcess }; });
ipcMain.handle('agent:running', () => ({ running: !!pyProcess }));
ipcMain.handle('agent:port', () => ({ port: API_PORT }));

app.whenReady().then(createWindow);
app.on('window-all-closed', () => {
  stopPython();
  if (process.platform !== 'darwin') app.quit();
});
app.on('before-quit', stopPython);
