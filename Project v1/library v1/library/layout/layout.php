<?php

/**
 * Description of layout
 *
 * @author durso
 */
namespace library\layout;
use library\layout\elements\element;
use library\layout\elements\script;

class layout extends element{
    private $container;
    private $elements;
    private $script;
    public function __construct(){
        $this->tag = "body";
    }
    public function render(){
        $this->openTag();
        foreach($this->children as $child){
            $this->html .= $child->render();
        }
        $this->html .= script::getScript();
        $this->closeTag();
        return $this->html;
    }
}
