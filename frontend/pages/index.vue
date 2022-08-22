<template>
  <client-only>
    <Video></Video>
  </client-only>
</template>

<script>
import Video from '../components/Video.vue';
export default {
    name: "IndexPage",
    components: { Video }
}
</script>
