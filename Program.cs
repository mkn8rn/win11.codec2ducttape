using System;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Reflection;
using NAudio.Wave;
using NAudio.Wasapi;
using NAudio.CoreAudioApi;

static class Codec2
{
    private const string Dll = "codec2";

    static Codec2()
    {
        NativeLibrary.SetDllImportResolver(typeof(Codec2).Assembly, (name, assembly, path) =>
        {
            if (name == Dll)
            {
                var exeDir = AppContext.BaseDirectory;
                var dllPath = Path.Combine(exeDir, "codec2.dll");
                if (File.Exists(dllPath))
                    return NativeLibrary.Load(dllPath);
            }
            return IntPtr.Zero;
        });
    }

    public static void Initialize() { } // Call to trigger static constructor

    [DllImport(Dll)] public static extern IntPtr codec2_create(int mode);
    [DllImport(Dll)] public static extern void   codec2_destroy(IntPtr codec);
    [DllImport(Dll)] public static extern int    codec2_samples_per_frame(IntPtr codec);
    [DllImport(Dll)] public static extern int    codec2_bytes_per_frame(IntPtr codec);
    [DllImport(Dll)] public static extern void   codec2_encode(IntPtr codec, byte[] bits, short[] speech);
    [DllImport(Dll)] public static extern void   codec2_decode(IntPtr codec, short[] speech, byte[] bits);
}

sealed class PitchShifter
{
    private float pitchFactor;
    private readonly float[] grainBuffer;
    private readonly int grainSize;
    private readonly int overlap;
    private float readPos;
    private int writePos;
    private readonly float[] window;
    private readonly float[] outputAccum;
    private int outputRead;

    public PitchShifter(int sampleRate, float semitones, int grainSize = 512)
    {
        this.grainSize = grainSize;
        this.overlap = grainSize / 2;
        this.pitchFactor = MathF.Pow(2f, semitones / 12f);
        
        grainBuffer = new float[grainSize * 4];
        outputAccum = new float[grainSize * 4];
        window = new float[grainSize];
        
        // Hann window
        for (int i = 0; i < grainSize; i++)
            window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / grainSize));
        
        readPos = 0;
        writePos = 0;
        outputRead = 0;
    }

    public void SetPitchSemitones(float semitones)
    {
        pitchFactor = MathF.Pow(2f, semitones / 12f);
    }

    public void Process(short[] samples)
    {
        if (MathF.Abs(pitchFactor - 1f) < 0.001f) return;

        for (int i = 0; i < samples.Length; i++)
        {
            // Write input to grain buffer
            grainBuffer[writePos] = samples[i] / 32768f;
            writePos = (writePos + 1) % grainBuffer.Length;

            // Read from grain buffer at shifted rate
            int readInt = (int)readPos;
            float frac = readPos - readInt;
            
            int idx0 = readInt % grainBuffer.Length;
            int idx1 = (readInt + 1) % grainBuffer.Length;
            
            // Linear interpolation
            float sample = grainBuffer[idx0] * (1f - frac) + grainBuffer[idx1] * frac;
            
            // Apply window for smoothing
            int windowPos = i % grainSize;
            float windowVal = window[windowPos];
            
            // Crossfade between grains
            if (windowPos < overlap)
            {
                float crossfade = windowPos / (float)overlap;
                sample *= crossfade;
            }
            else if (windowPos >= grainSize - overlap)
            {
                float crossfade = (grainSize - windowPos) / (float)overlap;
                sample *= crossfade;
            }

            samples[i] = (short)Math.Clamp(sample * 32768f, short.MinValue, short.MaxValue);
            
            // Advance read position at pitch-shifted rate
            readPos += pitchFactor;
            if (readPos >= grainBuffer.Length)
                readPos -= grainBuffer.Length;
        }
    }
}

sealed class FormantShifter
{
    private readonly float shiftFactor;
    private readonly int order;
    private readonly float[] lpcCoeffs;
    private readonly float[] history;

    public FormantShifter(float shiftRatio, int lpcOrder = 10)
    {
        this.shiftFactor = shiftRatio;
        this.order = lpcOrder;
        this.lpcCoeffs = new float[lpcOrder + 1];
        this.history = new float[lpcOrder];
    }

    public void Process(short[] samples)
    {
        if (MathF.Abs(shiftFactor - 1f) < 0.01f) return;

        var floatSamples = new float[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            floatSamples[i] = samples[i] / 32768f;

        ComputeLPC(floatSamples);
        ShiftFormants();
        ApplyLPC(floatSamples);

        for (int i = 0; i < samples.Length; i++)
            samples[i] = (short)Math.Clamp(floatSamples[i] * 32768f, short.MinValue, short.MaxValue);
    }

    private void ComputeLPC(float[] samples)
    {
        var autocorr = new float[order + 1];
        for (int i = 0; i <= order; i++)
        {
            for (int j = 0; j < samples.Length - i; j++)
                autocorr[i] += samples[j] * samples[j + i];
        }

        var err = autocorr[0];
        if (err < 1e-10f) return;

        for (int i = 0; i <= order; i++) lpcCoeffs[i] = 0;
        lpcCoeffs[0] = 1f;

        for (int i = 1; i <= order; i++)
        {
            float lambda = 0;
            for (int j = 0; j < i; j++)
                lambda -= lpcCoeffs[j] * autocorr[i - j];
            lambda /= err;

            for (int j = 0; j <= i / 2; j++)
            {
                float temp = lpcCoeffs[j] + lambda * lpcCoeffs[i - j];
                lpcCoeffs[i - j] += lambda * lpcCoeffs[j];
                lpcCoeffs[j] = temp;
            }

            err *= 1f - lambda * lambda;
        }
    }

    private void ShiftFormants()
    {
        for (int i = 1; i <= order; i++)
            lpcCoeffs[i] *= MathF.Pow(shiftFactor, i);
    }

    private void ApplyLPC(float[] samples)
    {
        Array.Clear(history);

        for (int i = 0; i < samples.Length; i++)
        {
            float excitation = samples[i];
            for (int j = 1; j <= order && i - j >= 0; j++)
                excitation += lpcCoeffs[j] * history[j - 1];

            for (int j = order - 1; j > 0; j--)
                history[j] = history[j - 1];
            history[0] = samples[i];

            samples[i] = excitation;
        }
    }
}

sealed class NoiseGate
{
    private readonly float threshold;
    private readonly float attackMs;
    private readonly float releaseMs;
    private readonly int sampleRate;
    private float envelope;
    private float gateGain;

    public NoiseGate(int sampleRate, float thresholdDb, float attackMs, float releaseMs)
    {
        this.sampleRate = sampleRate;
        this.threshold = MathF.Pow(10f, thresholdDb / 20f);
        this.attackMs = attackMs;
        this.releaseMs = releaseMs;
        this.envelope = 0f;
        this.gateGain = 0f;
    }

    public void Process(short[] samples)
    {
        float attackCoef = 1f - MathF.Exp(-1f / (sampleRate * attackMs / 1000f));
        float releaseCoef = 1f - MathF.Exp(-1f / (sampleRate * releaseMs / 1000f));

        for (int i = 0; i < samples.Length; i++)
        {
            float input = MathF.Abs(samples[i] / 32768f);

            if (input > envelope)
                envelope += attackCoef * (input - envelope);
            else
                envelope += releaseCoef * (input - envelope);

            float targetGain = envelope > threshold ? 1f : 0f;

            if (targetGain > gateGain)
                gateGain += attackCoef * (targetGain - gateGain);
            else
                gateGain += releaseCoef * (targetGain - gateGain);

            samples[i] = (short)(samples[i] * gateGain);
        }
    }
}

sealed class Compressor
{
    private readonly float threshold;
    private readonly float ratio;
    private readonly float attackMs;
    private readonly float releaseMs;
    private readonly float makeupGain;
    private readonly int sampleRate;
    private float envelope;

    public Compressor(int sampleRate, float thresholdDb, float ratio, float attackMs, float releaseMs, float makeupDb)
    {
        this.sampleRate = sampleRate;
        this.threshold = thresholdDb;
        this.ratio = ratio;
        this.attackMs = attackMs;
        this.releaseMs = releaseMs;
        this.makeupGain = MathF.Pow(10f, makeupDb / 20f);
        this.envelope = 0f;
    }

    public void Process(short[] samples)
    {
        float attackCoef = 1f - MathF.Exp(-1f / (sampleRate * attackMs / 1000f));
        float releaseCoef = 1f - MathF.Exp(-1f / (sampleRate * releaseMs / 1000f));

        for (int i = 0; i < samples.Length; i++)
        {
            float input = samples[i] / 32768f;
            float inputDb = 20f * MathF.Log10(MathF.Abs(input) + 1e-10f);

            if (inputDb > envelope)
                envelope += attackCoef * (inputDb - envelope);
            else
                envelope += releaseCoef * (inputDb - envelope);

            float gainReduction = 0f;
            if (envelope > threshold)
                gainReduction = (threshold - envelope) * (1f - 1f / ratio);

            float gain = MathF.Pow(10f, gainReduction / 20f) * makeupGain;
            samples[i] = (short)Math.Clamp(input * gain * 32768f, short.MinValue, short.MaxValue);
        }
    }
}

sealed class RobotVoice
{
    private readonly int sampleRate;
    private readonly float frequency;
    private float phase;

    public RobotVoice(int sampleRate, float frequency = 100f)
    {
        this.sampleRate = sampleRate;
        this.frequency = frequency;
        this.phase = 0f;
    }

    public void Process(short[] samples)
    {
        float phaseInc = 2f * MathF.PI * frequency / sampleRate;

        for (int i = 0; i < samples.Length; i++)
        {
            float carrier = MathF.Sign(MathF.Sin(phase));
            samples[i] = (short)(samples[i] * carrier);
            phase += phaseInc;
            if (phase > 2f * MathF.PI) phase -= 2f * MathF.PI;
        }
    }
}

sealed class Whisperizer
{
    private readonly Random rng = new();

    public void Process(short[] samples)
    {
        for (int i = 0; i < samples.Length; i++)
        {
            float envelope = MathF.Abs(samples[i] / 32768f);
            float noise = (float)(rng.NextDouble() * 2.0 - 1.0);
            samples[i] = (short)Math.Clamp(noise * envelope * 32768f, short.MinValue, short.MaxValue);
        }
    }
}

sealed class Distortion
{
    private readonly float drive;
    private readonly float mix;

    public Distortion(float drivePercent, float mixPercent)
    {
        // drive: 0-100, higher = more distortion
        this.drive = 1f + (drivePercent / 10f);
        this.mix = mixPercent / 100f;
    }

    public void Process(short[] samples)
    {
        for (int i = 0; i < samples.Length; i++)
        {
            float input = samples[i] / 32768f;
            // Soft clipping using tanh
            float distorted = MathF.Tanh(input * drive);
            float output = input * (1f - mix) + distorted * mix;
            samples[i] = (short)Math.Clamp(output * 32768f, short.MinValue, short.MaxValue);
        }
    }
}

static class Settings
{
    public static Dictionary<string, string> V = new();

    public static void LoadOrThrow(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(path);

        var d = new Dictionary<string, string>();

        foreach (var l in File.ReadAllLines(path))
        {
            var t = l.Trim();
            if (t.Length == 0 || t.StartsWith("#")) continue;

            var p = t.Split('=', 2);
            if (p.Length != 2)
                throw new FormatException($"Invalid line: {l}");

            d[p[0].Trim()] = p[1].Trim();
        }

        V = d;
    }

    public static void Save(string path)
    {
        var lines = V.OrderBy(k => k.Key)
                     .Select(kvp => $"{kvp.Key} = {kvp.Value}");
        File.WriteAllLines(path, lines);
    }

    public static string S(string k) =>
        V.TryGetValue(k, out var v)
            ? v
            : throw new InvalidOperationException($"Missing setting '{k}'");

    public static int I(string k) =>
        int.Parse(S(k), CultureInfo.InvariantCulture);

    public static float F(string k) =>
        float.Parse(S(k), CultureInfo.InvariantCulture);

    public static bool B(string k) =>
        bool.Parse(S(k));
}

sealed class BandPassFilter
{
    readonly float a0, a1, a2, b1, b2;
    float z1, z2;

    public BandPassFilter(int sampleRate, float lowHz, float highHz)
    {
        if (lowHz <= 0 || highHz <= lowHz || highHz >= sampleRate / 2)
            throw new ArgumentOutOfRangeException("Invalid band-pass frequencies");

        float fc = (lowHz + highHz) * 0.5f;
        float bw = highHz - lowHz;
        float q = fc / bw;

        float w0 = 2f * MathF.PI * fc / sampleRate;
        float alpha = MathF.Sin(w0) / (2f * q);
        float cosw0 = MathF.Cos(w0);

        float b0 = alpha;
        float b2 = -alpha;
        float a0n = 1f + alpha;
        float a1n = -2f * cosw0;
        float a2n = 1f - alpha;

        this.a0 = b0 / a0n;
        this.a1 = 0f;
        this.a2 = b2 / a0n;
        this.b1 = a1n / a0n;
        this.b2 = a2n / a0n;
    }

    public void Process(short[] s)
    {
        for (int i = 0; i < s.Length; i++)
        {
            float x = s[i] / 32768f;
            float y = a0 * x + a1 * z1 + a2 * z2 - b1 * z1 - b2 * z2;
            z2 = z1;
            z1 = y;
            s[i] = (short)Math.Clamp(y * 32768f, short.MinValue, short.MaxValue);
        }
    }
}

class Program
{
    static string GetProfilesDirectory()
    {
        var exeDir = AppContext.BaseDirectory;
        return Path.Combine(exeDir, "profiles");
    }

    static void EnsureProfilesDirectory()
    {
        var dir = GetProfilesDirectory();
        if (!Directory.Exists(dir))
            Directory.CreateDirectory(dir);
    }

    static string[] ListProfiles()
    {
        var dir = GetProfilesDirectory();
        if (!Directory.Exists(dir))
            return Array.Empty<string>();

        return Directory.GetFiles(dir, "*.txt")
                        .Select(Path.GetFileNameWithoutExtension)
                        .OrderBy(n => n != "default")  // default first
                        .ThenBy(n => n)
                        .ToArray();
    }

    static void SaveProfile(string name)
    {
        EnsureProfilesDirectory();
        var path = Path.Combine(GetProfilesDirectory(), $"{name}.txt");
        Settings.Save(path);
        Console.WriteLine($"Profile saved: {name}");
    }

    static bool TryLoadProfile(string name)
    {
        var path = Path.Combine(GetProfilesDirectory(), $"{name}.txt");
        if (!File.Exists(path))
            return false;
        Settings.LoadOrThrow(path);
        return true;
    }

    static void LoadProfile(string name)
    {
        if (!TryLoadProfile(name))
        {
            Console.WriteLine($"Profile not found: {name}");
            return;
        }
        Console.WriteLine($"Profile loaded: {name} (some changes require restart)");
    }

    static string SelectProfileOnStartup()
    {
        var profiles = ListProfiles();

        if (profiles.Length == 0)
        {
            Console.WriteLine("No profiles found in /profiles directory.");
            Console.WriteLine("Please create a profile .txt file in the profiles folder.");
            Environment.Exit(1);
        }

        Console.WriteLine("=== codec2ducttape ===");
        Console.WriteLine("Available profiles:");
        for (int i = 0; i < profiles.Length; i++)
            Console.WriteLine($"  [{i + 1}] {profiles[i]}");
        Console.WriteLine();

        while (true)
        {
            Console.Write("Select profile (number or name): ");
            var input = Console.ReadLine()?.Trim();

            if (string.IsNullOrEmpty(input))
                continue;

            // Try as number
            if (int.TryParse(input, out int num) && num >= 1 && num <= profiles.Length)
                return profiles[num - 1];

            // Try as name
            var match = profiles.FirstOrDefault(p => p.Equals(input, StringComparison.OrdinalIgnoreCase));
            if (match != null)
                return match;

            Console.WriteLine("Invalid selection. Try again.");
        }
    }

    static void PrintProfiles()
    {
        var profiles = ListProfiles();
        if (profiles.Length == 0)
        {
            Console.WriteLine("No profiles found in /profiles directory.");
            return;
        }
        Console.WriteLine("Available profiles:");
        foreach (var p in profiles)
            Console.WriteLine($"  {p}");
    }

    static int MapCodec2Mode(int bps) => bps switch
    {
        3200 => 0,  // CODEC2_MODE_3200
        2400 => 1,  // CODEC2_MODE_2400
        1600 => 2,  // CODEC2_MODE_1600
        1400 => 3,  // CODEC2_MODE_1400
        1300 => 4,  // CODEC2_MODE_1300
        1200 => 5,  // CODEC2_MODE_1200
        700  => 8,  // CODEC2_MODE_700C
        450  => 10, // CODEC2_MODE_450
        _ => throw new ArgumentOutOfRangeException(nameof(bps), $"Invalid codec2_mode: {bps}. Valid: 3200, 2400, 1600, 1400, 1300, 1200, 700, 450")
    };

    static void ApplyGain(short[] s, float g)
    {
        for (int i = 0; i < s.Length; i++)
            s[i] = (short)Math.Clamp(s[i] * g, short.MinValue, short.MaxValue);
    }

    static void ApplyLimiter(short[] s, float t)
    {
        int max = (int)(t * short.MaxValue);
        for (int i = 0; i < s.Length; i++)
            s[i] = (short)Math.Clamp(s[i], -max, max);
    }

    static void PrintAllSettings()
    {
        Console.WriteLine("=== Current Settings ===");
        foreach (var kvp in Settings.V.OrderBy(k => k.Key))
        {
            Console.WriteLine($"  {kvp.Key} = {kvp.Value}");
        }
        Console.WriteLine("========================");
    }

    static void Main()
    {
        var selectedProfile = SelectProfileOnStartup();
        if (!TryLoadProfile(selectedProfile))
        {
            Console.WriteLine($"ERROR: Failed to load profile '{selectedProfile}'");
            Console.WriteLine($"Profile path: {Path.Combine(GetProfilesDirectory(), $"{selectedProfile}.txt")}");
            return;
        }
        Console.WriteLine($"Loaded profile: {selectedProfile}");
        Console.WriteLine();

        int sampleRate = Settings.I("sample_rate");
        int mode = MapCodec2Mode(Settings.I("codec2_mode"));

        float micGain = Settings.F("mic_gain");
        float postGain = Settings.F("post_gain");

        bool bandpassEnabled = Settings.B("enable_bandpass");
        float bpLow = Settings.F("bandpass_low_hz");
        float bpHigh = Settings.F("bandpass_high_hz");

        bool limiter = Settings.B("enable_limiter");
        float limit = Settings.F("limiter_threshold");

        // Pitch shifting
        bool pitchEnabled = Settings.B("enable_pitch_shift");
        float pitchSemitones = Settings.F("pitch_semitones");

        // Formant shifting
        bool formantEnabled = Settings.B("enable_formant_shift");
        float formantRatio = Settings.F("formant_ratio");

        // Noise gate
        bool noiseGateEnabled = Settings.B("enable_noise_gate");
        float gateThreshold = Settings.F("noise_gate_threshold_db");
        float gateAttack = Settings.F("noise_gate_attack_ms");
        float gateRelease = Settings.F("noise_gate_release_ms");

        // Compressor
        bool compressorEnabled = Settings.B("enable_compressor");
        float compThreshold = Settings.F("compressor_threshold_db");
        float compRatio = Settings.F("compressor_ratio");
        float compAttack = Settings.F("compressor_attack_ms");
        float compRelease = Settings.F("compressor_release_ms");
        float compMakeup = Settings.F("compressor_makeup_db");

        // Voice effects
        bool robotEnabled = Settings.B("enable_robot_voice");
        float robotFreq = Settings.F("robot_frequency_hz");

        bool whisperEnabled = Settings.B("enable_whisper");

        bool codec2Enabled = Settings.B("enable_codec2");

        // Distortion
        bool distortionEnabled = Settings.B("enable_distortion");
        float distortionDrive = Settings.F("distortion_drive");
        float distortionMix = Settings.F("distortion_mix");

        BandPassFilter bandpass = bandpassEnabled
            ? new BandPassFilter(sampleRate, bpLow, bpHigh)
            : null;

        PitchShifter pitchShifter = pitchEnabled
            ? new PitchShifter(sampleRate, pitchSemitones)
            : null;

        FormantShifter formantShifter = formantEnabled
            ? new FormantShifter(formantRatio)
            : null;

        NoiseGate noiseGate = noiseGateEnabled
            ? new NoiseGate(sampleRate, gateThreshold, gateAttack, gateRelease)
            : null;

        Compressor compressor = compressorEnabled
            ? new Compressor(sampleRate, compThreshold, compRatio, compAttack, compRelease, compMakeup)
            : null;

        RobotVoice robotVoice = robotEnabled
            ? new RobotVoice(sampleRate, robotFreq)
            : null;

        Whisperizer whisperizer = whisperEnabled
            ? new Whisperizer()
            : null;

        Distortion distortion = distortionEnabled
            ? new Distortion(distortionDrive, distortionMix)
            : null;

        IntPtr codec = IntPtr.Zero;
        int spf = 160;
        int bpf = 8;

        if (codec2Enabled)
        {
            Codec2.Initialize(); // Ensure DLL resolver is set up
            codec = Codec2.codec2_create(mode);
            if (codec == IntPtr.Zero)
                throw new InvalidOperationException("codec2_create failed - check that codec2.dll is present and codec2_mode is valid (3200, 2400, 1600, 1400, 1300, 1200, 700)");

            spf = Codec2.codec2_samples_per_frame(codec);
            bpf = Codec2.codec2_bytes_per_frame(codec);
        }

        var pcmIn = new short[spf];
        var pcmOut = new short[spf];
        var bits = new byte[bpf];

        var buf = new short[spf * 4];
        int bufCount = 0;

        var mm = new MMDeviceEnumerator();
        var cap = mm.GetDefaultAudioEndpoint(DataFlow.Capture, Role.Console);
        var ren = mm.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active)
                    .First(d => d.FriendlyName.Contains(Settings.S("render_device")));

        var format = new WaveFormat(sampleRate, 16, 1);
        var capture = new WasapiCapture(cap) { WaveFormat = format };
        var provider = new BufferedWaveProvider(format) { DiscardOnBufferOverflow = true };

        var output = new WasapiOut(
            ren,
            AudioClientShareMode.Shared,
            false,
            Settings.I("latency_ms")
        );

        output.Init(provider);
        output.Play();

        capture.DataAvailable += (_, e) =>
        {
            int samples = e.BytesRecorded / 2;
            if (samples == 0) return;

            for (int i = 0; i < samples; i++)
            {
                buf[bufCount++] = BitConverter.ToInt16(e.Buffer, i * 2);

                if (bufCount >= spf)
                {
                    Array.Copy(buf, pcmIn, spf);
                    Array.Copy(buf, spf, buf, 0, bufCount - spf);
                    bufCount -= spf;

                    // Input processing chain
                    ApplyGain(pcmIn, micGain);
                    noiseGate?.Process(pcmIn);
                    bandpass?.Process(pcmIn);

                    // Voice effects
                    pitchShifter?.Process(pcmIn);
                    formantShifter?.Process(pcmIn);
                    robotVoice?.Process(pcmIn);
                    whisperizer?.Process(pcmIn);
                    distortion?.Process(pcmIn);

                    // Codec2 processing (optional)
                    if (codec2Enabled && codec != IntPtr.Zero)
                    {
                        Codec2.codec2_encode(codec, bits, pcmIn);
                        Codec2.codec2_decode(codec, pcmOut, bits);
                    }
                    else
                    {
                        Array.Copy(pcmIn, pcmOut, spf);
                    }

                    // Output processing chain
                    compressor?.Process(pcmOut);
                    ApplyGain(pcmOut, postGain);
                    if (limiter) ApplyLimiter(pcmOut, limit);

                    var ob = new byte[spf * 2];
                    Buffer.BlockCopy(pcmOut, 0, ob, 0, ob.Length);
                    provider.AddSamples(ob, 0, ob.Length);
                }
            }
        };

        capture.StartRecording();

        Console.WriteLine("Voice changer is running.");
        Console.WriteLine($"Capture format: {capture.WaveFormat}");
        Console.WriteLine($"Output format: {format}");
        Console.WriteLine($"Frame size: {spf} samples");
        Console.WriteLine("Commands: reload | status | profiles | save <name> | load <name> | help | ENTER = exit");

        while (true)
        {
            var cmd = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(cmd)) break;

            if (cmd.Equals("reload", StringComparison.OrdinalIgnoreCase))
            {
                TryLoadProfile(selectedProfile);
                
                micGain = Settings.F("mic_gain");
                postGain = Settings.F("post_gain");

                // Update pitch shifter
                if (Settings.B("enable_pitch_shift"))
                    pitchShifter?.SetPitchSemitones(Settings.F("pitch_semitones"));

                Console.WriteLine($"Profile '{selectedProfile}' reloaded (some changes require restart).");
            }
            else if (cmd.Equals("status", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine($"Current profile: {selectedProfile}");
                PrintAllSettings();
            }
            else if (cmd.Equals("help", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Available commands:");
                Console.WriteLine("  reload       - Reload current profile");
                Console.WriteLine("  status       - Show current profile and settings");
                Console.WriteLine("  profiles     - List available profiles");
                Console.WriteLine("  save <name>  - Save current settings to a profile");
                Console.WriteLine("  load <name>  - Load settings from a profile");
                Console.WriteLine("  help         - Show this help message");
                Console.WriteLine("  (enter)      - Exit the program");
            }
            else if (cmd.Equals("profiles", StringComparison.OrdinalIgnoreCase))
            {
                PrintProfiles();
            }
            else if (cmd.StartsWith("save ", StringComparison.OrdinalIgnoreCase))
            {
                var name = cmd.Substring(5).Trim();
                if (string.IsNullOrWhiteSpace(name))
                    Console.WriteLine("Usage: save <profile_name>");
                else
                    SaveProfile(name);
            }
            else if (cmd.StartsWith("load ", StringComparison.OrdinalIgnoreCase))
            {
                var name = cmd.Substring(5).Trim();
                if (string.IsNullOrWhiteSpace(name))
                {
                    Console.WriteLine("Usage: load <profile_name>");
                }
                else if (TryLoadProfile(name))
                {
                    selectedProfile = name;
                    
                    // Update runtime values
                    micGain = Settings.F("mic_gain");
                    postGain = Settings.F("post_gain");

                    if (Settings.B("enable_pitch_shift"))
                        pitchShifter?.SetPitchSemitones(Settings.F("pitch_semitones"));

                    Console.WriteLine($"Profile loaded: {name} (some changes require restart)");
                }
                else
                {
                    Console.WriteLine($"Profile not found: {name}");
                }
            }
        }

        capture.StopRecording();
        if (codec != IntPtr.Zero)
            Codec2.codec2_destroy(codec);
    }
}
