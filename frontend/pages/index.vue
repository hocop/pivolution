<template>
  <v-container>
  <client-only>
    <VideoSimple></VideoSimple>
  </client-only>
  <!-- <v-navigation-drawer
    absolute
    permanent
    right
  > -->

  </v-navigation-drawer>
  </v-container>
</template>

<script>
import VideoRTC from '../components/VideoRTC.vue';
import VideoSimple from '../components/VideoSimple.vue';
export default {
    name: "IndexPage",
    components: { VideoRTC, VideoSimple }
}
</script>
