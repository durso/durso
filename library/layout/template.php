<?php

/**
 * Description of template
 *
 * @author durso
 */
namespace library\layout;
use library\layout\elements\block;
use app\model\file;


class template{
    private $html = "";
    
    public function __construct($file) {
        $path = VIEW_PATH.DS.$file.".php"; 
        $string = file::read($path);
        $this->html = new block($string, array(), "div");
    }
    public function getTemplate(){
        return $this->html;
    }
    
}
