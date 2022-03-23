//! The life game renderer

use std::{ffi::CStr, mem::size_of, ptr::null};
use winit::{dpi::PhysicalSize, window::Window};

use ash::{
    extensions::{ext, khr},
    vk,
};

pub struct Renderer {
    allocation_callbacks: Option<vk::AllocationCallbacks>,
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_ext: Option<ext::DebugUtils>,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface_khr: khr::Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device: ash::Device,
    queue: vk::Queue,

    swapchain_khr: khr::Swapchain,
    surface_format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    present_mode: vk::PresentModeKHR,

    descriptor_pool: vk::DescriptorPool,

    life_width: u32,
    life_height: u32,
    life_image: vk::Image,
    life_memory: vk::DeviceMemory,
    life_image_views: [vk::ImageView; 2],
    life_set_layout: vk::DescriptorSetLayout,
    life_sets: [vk::DescriptorSet; 2],
    life_pipeline_layout: vk::PipelineLayout,
    life_pipeline: vk::Pipeline,

    life_init_staging_buffer: vk::Buffer,
    life_init_staging_memory: vk::DeviceMemory,
    life_init_command_pool: vk::CommandPool,
    life_init_command_buffer: vk::CommandBuffer,

    current_frame: usize,
    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; 2],
    fences: [vk::Fence; 2],
    available_semaphores: [vk::Semaphore; 2],
    presentable_semaphores: [vk::Semaphore; 2],

    render_set_layout: vk::DescriptorSetLayout,
    render_sets: [vk::DescriptorSet; 2],
    render_src_samplers: [vk::Sampler; 2],
    render_pass: vk::RenderPass,
    render_pipeline_layout: vk::PipelineLayout,
    render_pipeline: vk::Pipeline,

    extent: vk::Extent2D,
    new_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    image_count: usize,
    image_views: [vk::ImageView; 4],
    framebuffers: [vk::Framebuffer; 4],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct LifePushConstants {
    width: u32,
    height: u32,
}

impl Renderer {
    pub fn new(
        window: &Window,
        enable_debug: bool,
        life_size: (u32, u32),
    ) -> Self {
        unsafe {
            let entry = ash::Entry::load().unwrap();
            let extensions = entry.enumerate_instance_extension_properties(None).unwrap();
            let layers = entry.enumerate_instance_layer_properties().unwrap();
            let mut required_extensions = ash_window::enumerate_required_extensions(window)
                .unwrap()
                .iter()
                .map(|cs| cs.as_ptr())
                .collect::<Vec<_>>();
            let mut required_layers = vec![];

            let validation_enabled = if enable_debug {
                let debug_ext_name = ext::DebugUtils::name();
                let layer_name = b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8;

                match extensions
                    .iter()
                    .find(|e| CStr::from_ptr(e.extension_name.as_ptr()) == debug_ext_name)
                {
                    Some(_) => {
                        match layers.iter().find(|l| {
                            CStr::from_ptr(l.layer_name.as_ptr()) == CStr::from_ptr(layer_name)
                        }) {
                            Some(_) => {
                                required_extensions.push(debug_ext_name.as_ptr());
                                required_layers.push(layer_name);
                                true
                            }
                            None => false,
                        }
                    }
                    None => false,
                }
            } else {
                false
            };

            let app_info = vk::ApplicationInfo {
                api_version: vk::API_VERSION_1_0,
                ..Default::default()
            };

            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_extension_count: required_extensions.len() as _,
                pp_enabled_extension_names: required_extensions.as_ptr(),
                enabled_layer_count: required_layers.len() as _,
                pp_enabled_layer_names: required_layers.as_ptr(),
                ..Default::default()
            };

            let allocation_callbacks = None;

            let instance = entry
                .create_instance(&create_info, allocation_callbacks.as_ref())
                .unwrap();

            let (debug_utils_ext, debug_messenger) = if validation_enabled {
                let debug_utils_ext = ext::DebugUtils::new(&entry, &instance);
                let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
                    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    pfn_user_callback: Some(Self::debug_callback),
                    ..Default::default()
                };

                match debug_utils_ext
                    .create_debug_utils_messenger(&create_info, allocation_callbacks.as_ref())
                {
                    Ok(m) => (Some(debug_utils_ext), m),
                    Err(_) => (None, vk::DebugUtilsMessengerEXT::null()),
                }
            } else {
                (None, vk::DebugUtilsMessengerEXT::null())
            };

            let surface_khr = khr::Surface::new(&entry, &instance);

            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window,
                allocation_callbacks.as_ref(),
            )
            .unwrap();

            let (physical_device, physical_device_properties, queue_family_index) = instance
                .enumerate_physical_devices()
                .unwrap()
                .iter()
                .find_map(|&physical_device| {
                    let properties = instance.get_physical_device_properties(physical_device);
                    if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
                        return None;
                    }
                    let qf_properties =
                        instance.get_physical_device_queue_family_properties(physical_device);
                    match qf_properties
                        .iter()
                        .enumerate()
                        .position(|(index, properties)| {
                            properties
                                .queue_flags
                                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                                && surface_khr
                                    .get_physical_device_surface_support(
                                        physical_device,
                                        index as _,
                                        surface,
                                    )
                                    .unwrap()
                        }) {
                        Some(index) => Some((physical_device, properties, index as u32)),
                        None => None,
                    }
                })
                .unwrap();

            log::info!(
                "Pick physical device {}",
                CStr::from_ptr(physical_device_properties.device_name.as_ptr())
                    .to_str()
                    .unwrap()
            );

            let queue_priorities = [1.0];

            let queue_infos = [vk::DeviceQueueCreateInfo {
                queue_family_index,
                queue_count: queue_priorities.len() as _,
                p_queue_priorities: queue_priorities.as_ptr(),
                ..Default::default()
            }];

            let device_exts = [khr::Swapchain::name().as_ptr()];

            let device_features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: vk::TRUE,
                sampler_anisotropy: vk::TRUE,
                ..Default::default()
            };

            let create_info = vk::DeviceCreateInfo {
                queue_create_info_count: queue_infos.len() as _,
                p_queue_create_infos: queue_infos.as_ptr(),
                enabled_extension_count: device_exts.len() as _,
                pp_enabled_extension_names: device_exts.as_ptr(),
                p_enabled_features: &device_features,
                ..Default::default()
            };

            let device = instance
                .create_device(physical_device, &create_info, allocation_callbacks.as_ref())
                .unwrap();

            let queue = device.get_device_queue(queue_family_index, 0);

            let swapchain_khr = khr::Swapchain::new(&instance, &device);

            let mut surface_format = vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
            let surface_formats = surface_khr
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap();
            if !surface_formats.contains(&surface_format) {
                surface_format = surface_formats[0];
            }

            let mut present_mode = vk::PresentModeKHR::FIFO;
            if surface_khr
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap()
                .contains(&vk::PresentModeKHR::MAILBOX)
            {
                present_mode = vk::PresentModeKHR::MAILBOX;
            }

            let mut this = Self {
                allocation_callbacks,
                _entry: entry,
                instance,
                debug_utils_ext,
                debug_messenger,
                surface_khr,
                surface,
                physical_device,
                queue_family_index,
                device,
                queue,

                swapchain_khr,
                surface_format: surface_format.format,
                color_space: surface_format.color_space,
                present_mode,

                descriptor_pool: Default::default(),

                life_width: life_size.0,
                life_height: life_size.1,
                life_image: Default::default(),
                life_memory: Default::default(),
                life_image_views: Default::default(),
                life_set_layout: Default::default(),
                life_sets: Default::default(),
                life_pipeline_layout: Default::default(),
                life_pipeline: Default::default(),

                life_init_staging_buffer: Default::default(),
                life_init_staging_memory: Default::default(),
                life_init_command_pool: Default::default(),
                life_init_command_buffer: Default::default(),

                current_frame: 0,
                command_pool: Default::default(),
                command_buffers: Default::default(),
                fences: Default::default(),
                available_semaphores: Default::default(),
                presentable_semaphores: Default::default(),

                render_set_layout: Default::default(),
                render_sets: Default::default(),
                render_src_samplers: Default::default(),
                render_pass: Default::default(),
                render_pipeline_layout: Default::default(),
                render_pipeline: Default::default(),

                extent: Default::default(),
                new_extent: Default::default(),
                swapchain: Default::default(),
                image_count: 0,
                image_views: Default::default(),
                framebuffers: Default::default(),
            };

            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 4,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 2,
                },
            ];

            let create_info = vk::DescriptorPoolCreateInfo {
                max_sets: 4,
                pool_size_count: pool_sizes.len() as _,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            };

            this.descriptor_pool = this
                .device
                .create_descriptor_pool(&create_info, this.allocation_callbacks())
                .unwrap();

            this.create_life_game_resources();
            this.life_init();
            this.create_frame_resources();
            this.create_render_pass_resources();
            this.recreate_swapchain(vk::Extent2D {
                width: window.inner_size().width,
                height: window.inner_size().height,
            });
            this.new_extent = this.extent;

            this
        }
    }

    pub fn set_new_extent(&mut self, size: PhysicalSize<u32>) {
        self.new_extent = vk::Extent2D {
            width: size.width,
            height: size.height,
        };
    }

    pub fn render(&mut self) {
        unsafe {
            if self.new_extent.width != 0 && self.new_extent != self.extent {
                for i in [0, 1] {
                    if !self.device.get_fence_status(self.fences[i]).unwrap() {
                        return;
                    }
                }
                self.recreate_swapchain(self.new_extent);
            }

            let src_array_layer = self.current_frame as u32;
            let dst_array_layer = 1 - self.current_frame as u32;
            let life_set = self.life_sets[self.current_frame];
            let life_src_image_view = self.life_image_views[self.current_frame];
            let life_dst_image_view = self.life_image_views[1 - self.current_frame];
            let command_buffer = self.command_buffers[self.current_frame];
            let fence = self.fences[self.current_frame];
            let available_semaphore = self.available_semaphores[self.current_frame];
            let presentable_semaphore = self.presentable_semaphores[self.current_frame];
            let render_set = self.render_sets[self.current_frame];
            let src_sampler = self.render_src_samplers[self.current_frame];

            self.current_frame = 1 - self.current_frame;

            let (image_index, _) = self
                .swapchain_khr
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    available_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();

            let framebuffer = self.framebuffers[image_index as usize];

            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
                ..Default::default()
            };

            let device = &self.device;

            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();

            let image_infos = [
                vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: life_src_image_view,
                    image_layout: vk::ImageLayout::GENERAL,
                },
                vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: life_dst_image_view,
                    image_layout: vk::ImageLayout::GENERAL,
                },
                vk::DescriptorImageInfo {
                    sampler: src_sampler,
                    image_view: life_dst_image_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            ];

            let descriptor_writes = [
                vk::WriteDescriptorSet {
                    dst_set: life_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    p_image_info: &image_infos[0],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: life_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    p_image_info: &image_infos[1],
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: render_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: &image_infos[2],
                    ..Default::default()
                },
            ];

            device.update_descriptor_sets(&descriptor_writes, &[]);

            device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.life_pipeline,
            );

            let barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_READ,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: self.queue_family_index,
                dst_queue_family_index: self.queue_family_index,
                image: self.life_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: src_array_layer,
                    layer_count: 1,
                },
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[barrier],
            );

            let barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_READ,
                dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: self.queue_family_index,
                dst_queue_family_index: self.queue_family_index,
                image: self.life_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: dst_array_layer,
                    layer_count: 1,
                },
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[barrier],
            );

            let constants = LifePushConstants {
                width: self.life_width,
                height: self.life_height,
            };

            device.cmd_push_constants(
                command_buffer,
                self.life_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &constants as *const _ as _,
                    size_of::<LifePushConstants>(),
                ),
            );

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.life_pipeline_layout,
                0,
                &[life_set],
                &[],
            );

            device.cmd_dispatch(
                command_buffer,
                constants.width / 16,
                constants.height / 16,
                1,
            );

            let barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: self.queue_family_index,
                dst_queue_family_index: self.queue_family_index,
                image: self.life_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: dst_array_layer,
                    layer_count: 1,
                },
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[barrier],
            );

            let create_info = vk::RenderPassBeginInfo {
                render_pass: self.render_pass,
                framebuffer,
                render_area: vk::Rect2D {
                    extent: self.extent,
                    ..Default::default()
                },
                clear_value_count: 0,
                ..Default::default()
            };

            device.cmd_begin_render_pass(command_buffer, &create_info, vk::SubpassContents::INLINE);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as _,
                height: self.extent.height as _,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            let scissor = vk::Rect2D {
                extent: self.extent,
                ..Default::default()
            };

            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.render_pipeline,
            );

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.render_pipeline_layout,
                0,
                &[render_set],
                &[],
            );

            device.cmd_draw(command_buffer, 3, 1, 0, 0);

            device.cmd_end_render_pass(command_buffer);

            device.end_command_buffer(command_buffer).unwrap();

            let dst_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;

            let submit = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &available_semaphore,
                p_wait_dst_stage_mask: &dst_stage_mask,
                command_buffer_count: 1,
                p_command_buffers: &command_buffer,
                signal_semaphore_count: 1,
                p_signal_semaphores: &presentable_semaphore,
                ..Default::default()
            };

            device.reset_fences(&[fence]).unwrap();
            device.queue_submit(self.queue, &[submit], fence).unwrap();

            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: &presentable_semaphore,
                swapchain_count: 1,
                p_swapchains: &self.swapchain,
                p_image_indices: &image_index,
                ..Default::default()
            };

            self.swapchain_khr
                .queue_present(self.queue, &present_info)
                .unwrap();
        }
    }

    unsafe fn create_life_game_resources(&mut self) {
        let create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8_UNORM,
            extent: vk::Extent3D {
                width: self.life_width,
                height: self.life_height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 2,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &self.queue_family_index,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        self.life_image = self
            .device
            .create_image(&create_info, self.allocation_callbacks())
            .unwrap();
        let req = self.device.get_image_memory_requirements(self.life_image);
        self.life_memory = self
            .allocate_memory(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .unwrap();
        self.device
            .bind_image_memory(self.life_image, self.life_memory, 0)
            .unwrap();

        for i in [0, 1] {
            let create_info = vk::ImageViewCreateInfo {
                image: self.life_image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: vk::Format::R8_UNORM,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: i,
                    layer_count: 1,
                },
                ..Default::default()
            };

            self.life_image_views[i as usize] = self
                .device
                .create_image_view(&create_info, self.allocation_callbacks())
                .unwrap();
        }

        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as _,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        self.life_set_layout = self
            .device
            .create_descriptor_set_layout(&create_info, self.allocation_callbacks())
            .unwrap();

        let set_layouts = [self.life_set_layout, self.life_set_layout];
        let create_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 2,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };

        let sets = self.device.allocate_descriptor_sets(&create_info).unwrap();
        self.life_sets[0] = sets[0];
        self.life_sets[1] = sets[1];

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<LifePushConstants>() as _,
        };

        let create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &self.life_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            ..Default::default()
        };

        self.life_pipeline_layout = self
            .device
            .create_pipeline_layout(&create_info, self.allocation_callbacks())
            .unwrap();

        let module = self
            .create_shader_module(include_bytes!("life.comp.spv"))
            .unwrap();

        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module,
            p_name: b"main\0".as_ptr() as _,
            ..Default::default()
        };

        let create_info = vk::ComputePipelineCreateInfo {
            stage,
            layout: self.life_pipeline_layout,
            ..Default::default()
        };

        self.life_pipeline = self
            .device
            .create_compute_pipelines(
                Default::default(),
                &[create_info],
                self.allocation_callbacks(),
            )
            .unwrap()[0];

        self.device
            .destroy_shader_module(module, self.allocation_callbacks());
    }

    unsafe fn destroy_life_game_resources(&mut self) {
        self.device
            .destroy_pipeline(self.life_pipeline, self.allocation_callbacks());
        self.device
            .destroy_pipeline_layout(self.life_pipeline_layout, self.allocation_callbacks());
        self.device
            .destroy_descriptor_set_layout(self.life_set_layout, self.allocation_callbacks());
        for i in [0, 1] {
            self.device
                .destroy_image_view(self.life_image_views[i], self.allocation_callbacks());
        }
        self.device
            .free_memory(self.life_memory, self.allocation_callbacks());
        self.device
            .destroy_image(self.life_image, self.allocation_callbacks());
    }

    unsafe fn life_init(&mut self) {
        let width = self.life_width * (1800 - rand::random::<u8>() as u32) / (2055 - rand::random::<u8>() as u32);
        let height = self.life_height * 8 / 9;
        let size = (width * height) as _;

        let create_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &self.queue_family_index,
            ..Default::default()
        };

        self.life_init_staging_buffer = self
            .device
            .create_buffer(&create_info, self.allocation_callbacks())
            .unwrap();

        let req = self
            .device
            .get_buffer_memory_requirements(self.life_init_staging_buffer);
        self.life_init_staging_memory = self
            .allocate_memory(
                req,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();
        self.device
            .bind_buffer_memory(
                self.life_init_staging_buffer,
                self.life_init_staging_memory,
                0,
            )
            .unwrap();

        let data = self
            .device
            .map_memory(
                self.life_init_staging_memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();

        for b in std::slice::from_raw_parts_mut(data as *mut u8, size as _) {
            // *b = if rand::random::<bool>() { 255 } else { 0 };
            *b = 255;
        }

        self.device.unmap_memory(self.life_init_staging_memory);

        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT,
            queue_family_index: self.queue_family_index,
            ..Default::default()
        };

        self.life_init_command_pool = self
            .device
            .create_command_pool(&create_info, self.allocation_callbacks())
            .unwrap();

        let create_info = vk::CommandBufferAllocateInfo {
            command_pool: self.life_init_command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };

        self.life_init_command_buffer =
            self.device.allocate_command_buffers(&create_info).unwrap()[0];

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        let device = &self.device;
        let command_buffer = self.life_init_command_buffer;

        device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap();

        let src_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_queue_family_index: self.queue_family_index,
            dst_queue_family_index: self.queue_family_index,
            image: self.life_image,
            subresource_range: src_range,
            ..Default::default()
        };

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[barrier],
        );

        let barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: self.queue_family_index,
            dst_queue_family_index: self.queue_family_index,
            image: self.life_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 1,
                layer_count: 1,
            },
            ..Default::default()
        };

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[barrier],
        );

        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: ((self.life_width - width) / 2) as _,
                y: ((self.life_height - height) / 2) as _,
                z: 0,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };

        device.cmd_clear_color_image(
            command_buffer,
            self.life_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
            &[src_range],
        );

        device.cmd_copy_buffer_to_image(
            command_buffer,
            self.life_init_staging_buffer,
            self.life_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        let barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: self.queue_family_index,
            dst_queue_family_index: self.queue_family_index,
            image: self.life_image,
            subresource_range: src_range,
            ..Default::default()
        };

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::BY_REGION,
            &[],
            &[],
            &[barrier],
        );

        device.end_command_buffer(command_buffer).unwrap();

        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };

        device
            .queue_submit(self.queue, &[submit], vk::Fence::null())
            .unwrap();
    }

    unsafe fn destroy_life_init(&mut self) {
        self.device
            .destroy_command_pool(self.life_init_command_pool, self.allocation_callbacks());
        self.device
            .free_memory(self.life_init_staging_memory, self.allocation_callbacks());
        self.device
            .destroy_buffer(self.life_init_staging_buffer, self.allocation_callbacks());
    }

    unsafe fn create_frame_resources(&mut self) {
        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: self.queue_family_index,
            ..Default::default()
        };

        self.command_pool = self
            .device
            .create_command_pool(&create_info, self.allocation_callbacks())
            .unwrap();

        let create_info = vk::CommandBufferAllocateInfo {
            command_pool: self.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 2,
            ..Default::default()
        };

        let cbs = self.device.allocate_command_buffers(&create_info).unwrap();
        self.command_buffers[0] = cbs[0];
        self.command_buffers[1] = cbs[1];

        let fence_create_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };

        for i in [0, 1] {
            self.fences[i] = self
                .device
                .create_fence(&fence_create_info, self.allocation_callbacks())
                .unwrap();
            self.presentable_semaphores[i] = self
                .device
                .create_semaphore(&semaphore_create_info, self.allocation_callbacks())
                .unwrap();
            self.available_semaphores[i] = self
                .device
                .create_semaphore(&semaphore_create_info, self.allocation_callbacks())
                .unwrap();
        }
    }

    unsafe fn destroy_frame_resources(&mut self) {
        for semaphore in self.available_semaphores {
            self.device
                .destroy_semaphore(semaphore, self.allocation_callbacks());
        }
        for semaphore in self.presentable_semaphores {
            self.device
                .destroy_semaphore(semaphore, self.allocation_callbacks());
        }
        for fence in self.fences {
            self.device
                .destroy_fence(fence, self.allocation_callbacks());
        }
        self.device
            .destroy_command_pool(self.command_pool, self.allocation_callbacks());
    }

    unsafe fn create_render_pass_resources(&mut self) {
        let binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };

        let create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: 1,
            p_bindings: &binding,
            ..Default::default()
        };

        self.render_set_layout = self
            .device
            .create_descriptor_set_layout(&create_info, self.allocation_callbacks())
            .unwrap();

        let set_layouts = [self.render_set_layout, self.render_set_layout];
        let create_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 2,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };

        let sets = self.device.allocate_descriptor_sets(&create_info).unwrap();
        self.render_sets[0] = sets[0];
        self.render_sets[1] = sets[1];

        let create_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,
            compare_enable: vk::FALSE,
            min_lod: 0.0,
            max_lod: 1.0,
            ..Default::default()
        };

        for i in [0, 1] {
            self.render_src_samplers[i] = self
                .device
                .create_sampler(&create_info, self.allocation_callbacks())
                .unwrap();
        }

        let color_attachment = vk::AttachmentDescription {
            format: self.surface_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass_desc = vk::SubpassDescription {
            flags: vk::SubpassDescriptionFlags::empty(),
            input_attachment_count: 0,
            p_input_attachments: null(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref as _,
            p_resolve_attachments: null(),
            p_depth_stencil_attachment: null(),
            ..Default::default()
        };

        let create_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass_desc,
            dependency_count: 0,
            p_dependencies: null(),
            ..Default::default()
        };

        self.render_pass = self
            .device
            .create_render_pass(&create_info, self.allocation_callbacks())
            .unwrap();

        let create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &self.render_set_layout,
            ..Default::default()
        };

        self.render_pipeline_layout = self
            .device
            .create_pipeline_layout(&create_info, self.allocation_callbacks())
            .unwrap();

        let vert_shader = self
            .create_shader_module(include_bytes!("blit.vert.spv"))
            .unwrap();
        let frag_shader = self
            .create_shader_module(include_bytes!("blit.frag.spv"))
            .unwrap();

        let stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_shader,
                p_name: b"main\0".as_ptr() as _,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_shader,
                p_name: b"main\0".as_ptr() as _,
                ..Default::default()
            },
        ];

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            ..Default::default()
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };

        let attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            ..Default::default()
        };

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &attachment,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as _,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let create_info = vk::GraphicsPipelineCreateInfo {
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization_state,
            p_multisample_state: &multisample_state,
            p_depth_stencil_state: &depth_stencil_state,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state,
            layout: self.render_pipeline_layout,
            render_pass: self.render_pass,
            subpass: 0,
            ..Default::default()
        };

        self.render_pipeline = self
            .device
            .create_graphics_pipelines(
                Default::default(),
                &[create_info],
                self.allocation_callbacks(),
            )
            .unwrap()[0];

        self.device
            .destroy_shader_module(frag_shader, self.allocation_callbacks());
        self.device
            .destroy_shader_module(vert_shader, self.allocation_callbacks());
    }

    unsafe fn destroy_render_pass_resources(&mut self) {
        self.device
            .destroy_pipeline(self.render_pipeline, self.allocation_callbacks());
        self.device
            .destroy_pipeline_layout(self.render_pipeline_layout, self.allocation_callbacks());
        self.device
            .destroy_render_pass(self.render_pass, self.allocation_callbacks());
        for sampler in self.render_src_samplers {
            self.device
                .destroy_sampler(sampler, self.allocation_callbacks());
        }
        self.device
            .destroy_descriptor_set_layout(self.render_set_layout, self.allocation_callbacks());
    }

    unsafe fn recreate_swapchain(&mut self, extent: vk::Extent2D) {
        let surface_caps = self
            .surface_khr
            .get_physical_device_surface_capabilities(self.physical_device, self.surface)
            .unwrap();

        let create_info = vk::SwapchainCreateInfoKHR {
            surface: self.surface,
            min_image_count: 3,
            image_format: self.surface_format,
            image_color_space: self.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &self.queue_family_index,
            pre_transform: surface_caps.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: self.present_mode,
            clipped: vk::TRUE,
            old_swapchain: self.swapchain,
            ..Default::default()
        };

        let swapchain = self
            .swapchain_khr
            .create_swapchain(&create_info, self.allocation_callbacks())
            .unwrap();
        let images = self.swapchain_khr.get_swapchain_images(swapchain).unwrap();
        self.destroy_swapchain();
        self.extent = extent;
        self.swapchain = swapchain;
        self.image_count = images.len();

        for i in 0..images.len() {
            let create_info = vk::ImageViewCreateInfo {
                image: images[i],
                view_type: vk::ImageViewType::TYPE_2D,
                format: self.surface_format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            self.image_views[i] = self
                .device
                .create_image_view(&create_info, self.allocation_callbacks())
                .unwrap();
        }

        for i in 0..self.image_count {
            let create_info = vk::FramebufferCreateInfo {
                render_pass: self.render_pass,
                attachment_count: 1,
                p_attachments: &self.image_views[i],
                width: self.extent.width,
                height: self.extent.height,
                layers: 1,
                ..Default::default()
            };

            self.framebuffers[i] = self
                .device
                .create_framebuffer(&create_info, self.allocation_callbacks())
                .unwrap();
        }
    }

    unsafe fn destroy_swapchain(&mut self) {
        for i in 0..self.image_count {
            self.device
                .destroy_framebuffer(self.framebuffers[i], self.allocation_callbacks());
            self.device
                .destroy_image_view(self.image_views[i], self.allocation_callbacks());
        }
        self.swapchain_khr
            .destroy_swapchain(self.swapchain, self.allocation_callbacks());
    }

    fn allocation_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        self.allocation_callbacks.as_ref()
    }

    unsafe fn create_shader_module(&self, code: &[u8]) -> Result<vk::ShaderModule, vk::Result> {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len() as _,
            p_code: code.as_ptr() as _,
            ..Default::default()
        };

        self.device
            .create_shader_module(&create_info, self.allocation_callbacks())
    }

    unsafe fn allocate_memory(
        &self,
        req: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory, vk::Result> {
        let props = self
            .instance
            .get_physical_device_memory_properties(self.physical_device);
        let mut index = 0;
        while index < props.memory_type_count {
            if req.memory_type_bits & (1 << index) > 0 {
                if props.memory_types[index as usize]
                    .property_flags
                    .contains(flags)
                {
                    break;
                }
            }
            index += 1;
        }
        let create_info = vk::MemoryAllocateInfo {
            allocation_size: req.size,
            memory_type_index: index,
            ..Default::default()
        };
        self.device
            .allocate_memory(&create_info, self.allocation_callbacks())
    }

    unsafe extern "system" fn debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = *p_callback_data;

        let message = if callback_data.p_message.is_null() {
            std::borrow::Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };

        let level = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
            _ => unreachable!(),
        };

        log::log!(level, "{message_type:?} - {message}\n",);

        vk::FALSE
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.destroy_swapchain();
            self.destroy_render_pass_resources();
            self.destroy_frame_resources();
            self.destroy_life_init();
            self.destroy_life_game_resources();
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, self.allocation_callbacks());
            self.device.destroy_device(self.allocation_callbacks());
            self.surface_khr
                .destroy_surface(self.surface, self.allocation_callbacks());
            if let Some(debug_utils) = self.debug_utils_ext.as_ref() {
                debug_utils.destroy_debug_utils_messenger(
                    self.debug_messenger,
                    self.allocation_callbacks(),
                );
            }
            self.instance.destroy_instance(self.allocation_callbacks());
        }
    }
}
