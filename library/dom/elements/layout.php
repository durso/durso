<?php

/**
 * Description of layout
 *
 * @author durso
 */
namespace library\dom\elements;
use library\dom\elements\components\block;
use library\dom\elements\paired;
use library\dom\object;

class layout extends paired{
    private $layout = null;

    public function __construct(){
        parent::__construct();
        $this->tag = "body";
        $this->layout = new block("div");
        $this->addComponent($this->layout);
    }
    public function setLayoutType($type){
        if($type == 'fluid'){
            $this->fluidLayout();
        } else {
            $this->fixedLayout();
        }
    } 
    private function fixedLayout(){
        $this->layout->addClass("container");
    }
    private function fluidLayout(){
        $this->layout->addClass("container-fluid");
    }
    public function addComponent(object $component) {
        if(!$this->node->hasChild()){
            parent::addComponent($component);
        } else {
            $this->node->getChild(0)->getValue()->addComponent($component);
        }
    }
}
