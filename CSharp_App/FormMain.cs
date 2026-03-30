using OpenCvSharp;
using OpenCvSharp.Extensions;
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

        // Python環境設定（実行環境に合わせて調整してください）
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
            // ボタンイベント等はデザイナ側、またはここで紐付け
            btnStart.Click += (s, e) => StartStreaming();
            btnStop.Click += (s, e) => StopStreaming();
        }

        private void SetupTimer()
        {
            _videoTimer = new Timer { Interval = 1000 / 15 }; // 15 FPS
            _videoTimer.Tick += VideoTimer_Tick;
        }

        #region 初期化・終了処理
        private void FormMain_Load(object sender, EventArgs e)
        {
            label3.Text = "YOLOモデルのバージョン情報を取得中...";
            CheckPythonModel();
        }

        private void FormMain_FormClosing(object sender, FormClosingEventArgs e)
        {
            StopStreaming();
            _currentFrame?.Dispose();
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

                    if (process.ExitCode == 0)
                    {
                        label3.Text = $"Model Status: OK\n{output.Trim()}";
                    }
                    else
                    {
                        label3.Text = "Model Load Error: " + error;
                    }
                }
            }
            catch (Exception ex)
            {
                label3.Text = "Python連携エラー: " + ex.Message;
            }
        }
        #endregion

        #region ビデオ制御ロジック
        private void StartStreaming()
        {
            if (_isStreaming) return;

            // RTSPまたはカメラの読み込み (0はWebカメラ)
            _capture = new VideoCapture(0); 
            if (!_capture.IsOpened())
            {
                MessageBox.Show("カメラ/ストリームを開けませんでした。");
                return;
            }

            _currentFrame = new Mat();
            _isStreaming = true;
            _videoTimer.Start();
            labelStatus.Text = "Streaming: Active";
        }

        private void StopStreaming()
        {
            _videoTimer.Stop();
            _isStreaming = false;
            lock (_captureLock)
            {
                _capture?.Release();
                _capture = null;
            }
            labelStatus.Text = "Streaming: Stopped";
        }

        private void VideoTimer_Tick(object sender, EventArgs e)
        {
            lock (_captureLock)
            {
                if (_capture == null || !_capture.IsOpened()) return;

                _capture.Read(_currentFrame);
                if (_currentFrame.Empty()) return;

                // PictureBoxに表示
                pictureBoxVideo.Image?.Dispose();
                pictureBoxVideo.Image = _currentFrame.ToBitmap();
            }

            // 5フレームに1回など、負荷を調整して解析を実行
            Task.Run(() => RunInference());
        }
        #endregion

        #region Pythonプロセス連携
        /// <summary>
        /// 現在のフレームを一時保存し、Pythonに渡して解析結果を受け取ります。
        /// </summary>
        private void RunInference()
        {
            try
            {
                string tempImg = Path.Combine(ScriptDir, "temp_frame.jpg");
                lock (_captureLock)
                {
                    if (_currentFrame == null || _currentFrame.Empty()) return;
                    _currentFrame.SaveImage(tempImg);
                }

                var psi = CreateProcessStartInfo($"\"{ScriptPath}\" --predict \"{tempImg}\"");
                using (var process = Process.Start(psi))
                {
                    string result = process.StandardOutput.ReadToEnd();
                    process.WaitForExit();

                    // 解析結果をGUIに反映
                    this.Invoke(new Action(() => {
                        labelResult.Text = "解析値: " + result.Trim();
                    }));
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine("Inference Error: " + ex.Message);
            }
        }

        private ProcessStartInfo CreateProcessStartInfo(string arguments)
        {
            return new ProcessStartInfo
            {
                FileName = PythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8
            };
        }
        #endregion
    }
}
