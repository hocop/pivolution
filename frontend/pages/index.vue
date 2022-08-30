<template>
  <client-only>
    <!-- <VideoRTC></VideoRTC> -->
    <VideoSimple></VideoSimple>
  </client-only>
</template>

<script>
import VideoRTC from '../components/VideoRTC.vue';
import VideoSimple from '../components/VideoSimple.vue';
export default {
    name: "IndexPage",
    components: { VideoRTC, VideoSimple }
}
</script>
