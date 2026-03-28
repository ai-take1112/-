using OpenCvSharp;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Timer = System.Windows.Forms.Timer;

namespace AnalogReader
{
    /// <summary>
    /// アナログ計器読み取りシステム メイン画面
    /// RTSPストリーム再生とPython(YOLO)による画像解析を制御します。
    /// </summary>
    public partial class FormMain : Form
    {
        #region フィールド・プロパティ
        private VideoCapture _capture;
        private Mat _currentFrame;
        private Timer _videoTimer;
        private readonly object _captureLock = new object();
        private bool _isStreaming = false;

        // Python環境設定（環境に合わせて変更可能）
        private const string PythonExe = @"C:\Users\小田桐健\AppData\Local\Programs\Python\Python310\python.exe";
        private const string ScriptDir = @"C:\Users\小田桐健\Desktop\pyhon_image_trian";
        private string ScriptPath => Path.Combine(ScriptDir, "voltmeter_read.py");
        #endregion

        public FormMain()
        {
            InitializeComponent();
            SetupEvents();
            SetupTimer();
        }

        private void SetupEvents()
        {
            this.Load += FormMain_Load;
            this.FormClosing += FormMain_FormClosing;
        }

        private void SetupTimer()
        {
            _videoTimer = new Timer { Interval = 1000 / 15 }; // 15 FPS
            _videoTimer.Tick += VideoTimer_Tick;
        }

        #region 初期化処理
        private void FormMain_Load(object sender, EventArgs e)
        {
            label3.Text = "YOLOモデルのバージョン情報を取得中...";
            CheckPythonModel();
        }

        /// <summary>
        /// Pythonスクリプトをテストモードで起動し、YOLOモデルの読み込みを確認します。
        /// </summary>
        private void CheckPythonModel()
        {
            try
            {
                var psi = CreateProcessStartInfo($"\"{ScriptPath}\" --test");
                using (var process = Process.Start(psi))
                {
                    string output = process.StandardOutput.ReadToEnd();
                    string error = process.StandardError.ReadToEnd();
                    process.WaitForExit();

                    label3.
