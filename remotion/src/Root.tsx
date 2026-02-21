import React from "react";
import { Composition, Folder } from "remotion";

import { HomeHeroVideo, homeHeroVideoProps } from "./HomeHeroVideo";
import { HomeHeroVideoZh, homeHeroVideoZhProps } from "./HomeHeroVideoZh";

const fps = 30;

export const RemotionRoot: React.FC = () => {
  return (
    <Folder name="Homepage">
      <Composition
        id="HomeHeroVideo"
        component={HomeHeroVideo}
        durationInFrames={30 * fps}
        fps={fps}
        width={1280}
        height={720}
        defaultProps={homeHeroVideoProps}
      />
      <Composition
        id="HomeHeroVideoZh"
        component={HomeHeroVideoZh}
        durationInFrames={30 * fps}
        fps={fps}
        width={1280}
        height={720}
        defaultProps={homeHeroVideoZhProps}
      />
    </Folder>
  );
};
