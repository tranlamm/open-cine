import {Config} from '@remotion/cli/config';

// The scribble layer is rendered with transparency (render.mjs writes PNG frames itself).
Config.setVideoImageFormat('png');
Config.setPixelFormat('yuva444p10le');
Config.setCodec('prores');
Config.setProResProfile('4444');
