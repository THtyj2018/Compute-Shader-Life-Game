mod renderer;

use renderer::Renderer;

use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, WindowBuilder},
};

fn main() {
    let params = match EnvParams::from_env() {
        Ok(Some(params)) => params,
        Ok(None) => return,
        Err(e) => {
            println!("Error: Invalid input; {}", e.to_string());
            println!("Use '-h' or '--help' for more information");
            return;
        }
    };
    init_log(log::LevelFilter::Info, "lifegame.log");

    let event_loop = EventLoop::new();
    let window = if params.fullscreen {
        WindowBuilder::new().with_fullscreen(Some(Fullscreen::Borderless(None)))
    } else {
        WindowBuilder::new().with_inner_size(PhysicalSize::new(params.width, params.height))
    }
    .with_title("Conway Life Game")
    .build(&event_loop)
    .unwrap();

    let mut renderer = Renderer::new(&window, true, (params.back_width, params.back_height));

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(size) => renderer.set_new_extent(size),
                _ => (),
            },
            Event::MainEventsCleared => {
                if window.inner_size().width == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                } else {
                    renderer.render();
                }
            }
            _ => (),
        }
    });
}

struct EnvParams {
    fullscreen: bool,
    width: u32,
    height: u32,
    back_width: u32,
    back_height: u32,
}

impl EnvParams {
    fn from_env() -> Result<Option<Self>, anyhow::Error> {
        let args = std::env::args();

        let mut this = Self {
            fullscreen: false,
            width: 1280,
            height: 720,
            back_width: 2560,
            back_height: 1440,
        };

        for arg in args {
            if arg == "-h" || arg == "--help" {
                println!("Parameters:");
                println!("    -h, --help:         get this help information");
                println!("    -f, --fullscreen:   set to fullscreen window");
                println!("    -wN:    set the window width to N (> 0)");
                println!("    -hN:    set the window height to N (> 0)");
                println!("    -bwN:   set the life game height to N (> 0, = 16M)");
                println!("    -bhN:   set the life game height to N (> 0, = 16M)");
                return Ok(None);
            } else if arg == "-f" || arg == "--fullscreen" {
                this.fullscreen = true;
            } else if arg.starts_with("-w") {
                this.width = arg.trim_start_matches("-w").parse()?;
            } else if arg.starts_with("-h") {
                this.height = arg.trim_start_matches("-h").parse()?;
            } else if arg.starts_with("-bw") {
                this.back_width = arg.trim_start_matches("-bw").parse()?;
            } else if arg.starts_with("-bh") {
                this.back_height = arg.trim_start_matches("-bh").parse()?;
            }
        }

        if this.width == 0 || this.height == 0 {
            return Err(anyhow::Error::msg(
                "width and height must be greater than 0",
            ));
        }

        if this.back_width == 0 || this.back_height == 0 {
            return Err(anyhow::Error::msg("backbuffer size must be greater than 0"));
        }

        if this.back_width % 16 > 0 || this.back_height % 16 > 0 {
            return Err(anyhow::Error::msg(
                "backbuffer dimensions must be divisible by 16",
            ));
        }

        Ok(Some(this))
    }
}

fn init_log(level: log::LevelFilter, filepath: &str) {
    let encoder = log4rs::encode::pattern::PatternEncoder::new("{d} - {l} - {t} - {m}{n}");
    let file_appender = log4rs::append::file::FileAppender::builder()
        .encoder(Box::new(encoder))
        .append(false)
        .build(filepath)
        .unwrap();
    let appender = log4rs::config::Appender::builder().build("file", Box::new(file_appender));
    let logger = log4rs::config::Logger::builder()
        .appender("file")
        .additive(false)
        .build("konrf", level);
    let root = log4rs::config::Root::builder()
        .appender("file")
        .build(level);
    let config = log4rs::config::Config::builder()
        .appender(appender)
        .logger(logger)
        .build(root)
        .unwrap();
    log4rs::init_config(config).unwrap();
}
