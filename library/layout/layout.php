<?php

/**
 * Description of layout
 *
 * @author durso
 */
class layout {
    protected $tag;
    protected $attributes = array();
    protected $elements = array();
    protected $components = array();
    
    public function __construct(){
        $this->tag = "body";
    }
}
