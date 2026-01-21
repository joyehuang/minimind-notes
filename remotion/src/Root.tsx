import React from "react";
import { Composition, Folder } from "remotion";

import { HomeHeroVideo, homeHeroVideoProps } from "./HomeHeroVideo";

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
    </Folder>
  );
};
